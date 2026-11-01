"""Action chassis: gather → reason → writeback as a three-node graph (graph ``action``).

```
[gather]     code collects the inputs (port). No inputs → the run fails, it does not guess.
[reason]     the model decides, at the manifest's level:
               structured     one structured-output call, no tools
               bounded-react  a ReAct loop over the skill's read tools, at most 8 rounds
               scoped-agent   a deep agent scoped to the skill (read tools, files, constrained subagent)
[writeback]  code stores the result (port), after it validated against output_schema.
```

Rules that make the lane predictable:

* writing is never an LLM tool here: an action whose skill reaches a ``write`` tool does not build;
* the node layout never depends on the manifest, so one schema-only graph reads every action thread;
* the skill and prompts are pinned when gather runs and stay in thread state (``asset_versions``);
* the same event rows as the agent lane, with ``kind: action``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, Optional, TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from wigtn_agents.actions.ports import (
    GatherRequest,
    WritebackRequest,
    call_port,
    gatherer_for,
    writeback_for,
)
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import Graph, RunContext
from wigtn_agents.manifest import ActionManifest
from wigtn_agents.middleware import (
    EventRecorderMiddleware,
    EventSink,
    JsonlSink,
    RunStamp,
    build_event,
    compose_prompt,
    pin_assets,
)
from wigtn_agents.middleware.events import _configurable, usage_of
from wigtn_agents.middleware.extra import MiddlewareSpec, build_extra_middleware
from wigtn_agents.middleware.progress import emit_progress
from wigtn_agents.middleware.scope_guard import SCOPE_PARAMS
from wigtn_agents.models import ModelRegistry, ResolvedModel, build_model
from wigtn_agents.presets import loop_stack, preset_permissions
from wigtn_agents.tools.gateway import ToolClients, build_tools, schema_model

# Tool rounds the model may take before its final (structured) answer. Enforced as a model-call
# limit (rounds + 1) that ends the loop cleanly; the recursion limit is only a backstop, because
# every middleware hook node also counts as a graph step.
BOUNDED_REACT_ROUNDS: Final = 8
SCOPED_AGENT_ROUNDS: Final = 24
STEPS_PER_ROUND_BACKSTOP: Final = 25
OUTPUT_MODEL_NAME: Final = "ActionResult"
ACTION_NODES: Final = ("gather", "reason", "writeback")


class ActionState(TypedDict, total=False):
    """Thread state of one action run. The client sends ``{"input": {...}}``."""

    input: dict[str, Any]
    gathered: dict[str, Any]
    result: dict[str, Any]
    writeback: dict[str, Any]
    asset_versions: dict[str, str]
    run_meta: dict[str, Any]


def output_model(action: ActionManifest) -> type[BaseModel]:
    """``output_schema`` as the model the reason step must produce."""
    model = schema_model(OUTPUT_MODEL_NAME, action.output_schema)
    model.__doc__ = f"Result of action {action.agent_id}."
    return model


def _render(gathered: dict[str, Any]) -> str:
    if not gathered:
        return "작업을 수행하세요."
    return "\n\n".join(
        f"[{key}]\n{json.dumps(value, ensure_ascii=False, indent=2, default=str)}" for key, value in gathered.items()
    )


def _scoped_args(name: str, args: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """Scope arguments come from the run context, never from the input (same rule as ScopeGuard)."""
    scoped = dict(args)
    for arg_name, ctx_key in SCOPE_PARAMS.get(name, ()):
        canonical = ctx.get(ctx_key)
        if canonical is None:
            raise LookupError(f"스코프 정본 없음 ({name}.{arg_name}): run context에 {ctx_key}가 필요합니다")
        scoped[arg_name] = canonical
    return scoped


class GatherToolError(RuntimeError):
    """A default-plan tool failed; the action stops instead of reasoning over an error text."""


async def gather_default_plan(
    request: GatherRequest,
    plan: Sequence[str],
    on_call: Optional[Callable[[str, str, int], None]] = None,
) -> dict[str, Any]:
    """Call the skill's ``default_plan`` tools in order with the input as arguments.

    Tools are invoked as tool calls so a failure comes back as an error ``ToolMessage`` (the
    tools handle their own errors for the model loop); here that failure ends the run. Each call
    is reported to ``on_call(name, status, latency_ms)`` for the run record.
    """
    gathered: dict[str, Any] = {}
    for index, name in enumerate(plan):
        tool: BaseTool = request.tools[name]
        fields = set(getattr(tool.args_schema, "model_fields", {}))
        args = _scoped_args(name, {k: v for k, v in request.input.items() if k in fields}, request.ctx)
        started = time.perf_counter()
        message = await tool.ainvoke({"type": "tool_call", "id": f"gather-{index}", "name": name, "args": args})
        latency_ms = int((time.perf_counter() - started) * 1000)
        failed = isinstance(message, ToolMessage) and message.status == "error"
        if on_call is not None:
            on_call(name, "error" if failed else "ok", latency_ms)
        if failed:
            raise GatherToolError(f"gather 도구 실패: {name}")
        gathered[name] = message.artifact if getattr(message, "artifact", None) is not None else _content(message)
    return gathered


def _content(message: Any) -> Any:
    """A tool's return value back from its ``ToolMessage`` (JSON when it was a dict)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


class ActionRecorder:
    """Lifecycle rows of one action run (the inner tool loop records its own calls)."""

    def __init__(self, sink: EventSink, stamp: RunStamp) -> None:
        self._sink = sink
        self.stamp = stamp

    def emit(self, event_type: str, versions: dict[str, str], **fields: Any) -> None:
        """One row."""
        self._sink.append(build_event(self.stamp, event_type, asset_versions=versions, **fields))


def _step(name: str, status: str) -> None:
    emit_progress({"type": "progress", "kind": "step", "step": name, "status": status})


Node = Callable[[ActionState], Awaitable[dict[str, Any]]]


def _guarded(name: str, recorder: ActionRecorder, body: Node) -> Node:
    """Progress markers around a node; a failure is recorded with the class name only."""

    async def node(state: ActionState) -> dict[str, Any]:
        _step(name, "running")
        try:
            update = await body(state)
        except GraphBubbleUp:
            raise  # an interrupt raised by a port is a pause, not a failure
        except asyncio.CancelledError:
            recorder.emit("run.end", state.get("asset_versions") or {}, status="cancelled", step=name)
            raise
        except BaseException as exc:
            _step(name, "error")
            versions = state.get("asset_versions") or {}
            recorder.emit("run.error", versions, step=name, error_class=type(exc).__name__)
            recorder.emit("run.end", versions, status="error")
            raise
        _step(name, "completed")
        return update

    return node


def build_action_graph(
    action: ActionManifest,
    catalog: Catalog,
    ctx: RunContext,
    *,
    resolved: ResolvedModel,
    model: Optional[BaseChatModel] = None,
    model_registry: Optional[ModelRegistry] = None,
    clients: Optional[ToolClients] = None,
    sink: Optional[EventSink] = None,
    extra_names: tuple[str, ...] = (),
    inner_middleware: Sequence[AgentMiddleware] = (),
) -> Graph:
    """Compile the chassis for one action and one run context (no checkpointer/store)."""
    clients = clients or ToolClients()
    sink = sink if sink is not None else JsonlSink()
    tool_assets = catalog.tools_for_manifest(action)
    writes = [a.ref for a in tool_assets if a.typed_body.side_effect == "write"]  # type: ignore[attr-defined]
    if writes:
        raise ValueError(f"{action.agent_id}: action의 도구는 읽기 전용입니다. 쓰기는 writeback port로 ({writes})")
    tools = {t.name: t for t in build_tools(action, catalog, clients)}
    skill = catalog.resolve(action.skill)
    plan = [catalog.resolve(ref).tool_name() for ref in skill.typed_body.default_plan]  # type: ignore[attr-defined]
    gatherer = None if action.gather == "default-plan" else gatherer_for(action.gather)
    writer = writeback_for(action.writeback)
    result_model = output_model(action)
    if action.level == "structured" and (extra_names or inner_middleware):
        raise ValueError(
            f"{action.agent_id}: structured 수준에는 도구 루프가 없어 extra 미들웨어를 붙일 수 없습니다 "
            f"({list(extra_names)}). bounded-react 또는 scoped-agent로 바꾸세요"
        )
    recorder = ActionRecorder(sink, RunStamp.for_manifest(action, ctx, resolved, extra_middleware=extra_names))
    extra = [*build_extra_middleware(extra_names, MiddlewareSpec(action, ctx)), *inner_middleware]

    def chat_model() -> BaseChatModel:
        return model or build_model(resolved.alias, registry=model_registry, **action.model.params())

    async def gather(state: ActionState) -> dict[str, Any]:
        versions = state.get("asset_versions") or pin_assets(action, catalog)
        recorder.emit("run.start", versions, level=action.level)
        request = GatherRequest(action=action, input=dict(state.get("input") or {}), ctx=ctx, tools=tools)
        if gatherer is None:

            def record(name: str, status: str, latency_ms: int) -> None:
                tool = {"name": name, "allowed": True, "denied_by": None, "status": status, "latency_ms": latency_ms}
                recorder.emit("tool.call", versions, step="gather", tool=tool)

            gathered = await gather_default_plan(request, plan, record)
        else:
            gathered = await call_port(gatherer, request)
        meta = {
            "agent_id": action.agent_id,
            "manifest_version": action.metadata.version,
            "model_alias": resolved.alias,
            "model_id": resolved.model_id,
            "image_version": "dev",
        }
        return {"asset_versions": versions, "run_meta": meta, "gathered": gathered}

    async def reason(state: ActionState) -> dict[str, Any]:
        versions = state.get("asset_versions") or {}
        system = compose_prompt(action, catalog, versions)
        human = _render(state.get("gathered") or {})
        if action.level == "structured":
            structured = chat_model().with_structured_output(result_model, include_raw=True)
            out = await structured.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
            raw = out.get("raw") if isinstance(out, dict) else None
            recorder.emit("llm.call", versions, tool_call_count=0, usage=usage_of(raw))
            if isinstance(out, dict) and out.get("parsing_error") is not None:
                raise ValueError(f"구조화 출력 파싱 실패: {type(out['parsing_error']).__name__}")
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if not isinstance(parsed, result_model):
                raise ValueError("reason 단계가 output_schema 결과를 내지 못했습니다")
            return {"result": parsed.model_dump()}

        loop_recorder = EventRecorderMiddleware(sink, recorder.stamp, asset_versions=versions, lifecycle=False)
        middleware = loop_stack(
            action.harness_preset,
            recorder=loop_recorder,
            allowed_tools=set(tools),
            extra=extra,
            static_context=dict(ctx),
            output_validator=False,
            model_call_limit=(BOUNDED_REACT_ROUNDS if action.level == "bounded-react" else SCOPED_AGENT_ROUNDS) + 1,
        )
        if action.level == "bounded-react":
            from langchain.agents import create_agent

            agent: Any = create_agent(
                model=chat_model(),
                tools=list(tools.values()),
                system_prompt=system,
                response_format=result_model,
                middleware=middleware,
            )
            rounds = BOUNDED_REACT_ROUNDS
        else:
            from deepagents import create_deep_agent

            from wigtn_agents.assembly import constrained_general_purpose, subagent_stamp

            options: dict[str, Any] = {}
            permissions = preset_permissions(action.harness_preset)
            if permissions:
                options["permissions"] = permissions
            agent = create_deep_agent(
                model=chat_model(),
                tools=list(tools.values()),
                system_prompt=system,
                middleware=middleware,
                subagents=[
                    constrained_general_purpose(
                        recorder=EventRecorderMiddleware(
                            sink, subagent_stamp(recorder.stamp, "general-purpose"), asset_versions=versions
                        ),
                        preset=action.harness_preset,
                    )
                ],
                response_format=result_model,
                name=f"{action.agent_id}-reason",
                **options,
            )
            rounds = SCOPED_AGENT_ROUNDS
        outcome = await agent.ainvoke(
            {"messages": [HumanMessage(content=human)]},
            config={"recursion_limit": (rounds + 1) * STEPS_PER_ROUND_BACKSTOP},
        )
        parsed = outcome.get("structured_response")
        if not isinstance(parsed, result_model):
            raise ValueError(
                f"{action.level} reason 단계가 output_schema 결과를 내지 못했습니다 (도구 라운드 상한 {rounds}회)"
            )
        return {"result": parsed.model_dump()}

    async def writeback(state: ActionState) -> dict[str, Any]:
        versions = state.get("asset_versions") or {}
        payload = result_model.model_validate(state.get("result") or {}).model_dump()
        request = WritebackRequest(action=action, result=payload, ctx=ctx, run_id=_configurable().get("run_id"))
        report = await call_port(writer, request)
        recorder.emit("run.end", versions, status="ok", writeback=report.get("state"))
        return {"writeback": report}

    graph: StateGraph[ActionState] = StateGraph(ActionState)
    for name, body in (("gather", gather), ("reason", reason), ("writeback", writeback)):
        graph.add_node(name, _guarded(name, recorder, body))  # type: ignore[call-overload]
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "reason")
    graph.add_edge("reason", "writeback")
    graph.add_edge("writeback", END)
    return graph.compile(name=action.agent_id)


def schema_only_action_graph(reason: str) -> Graph:
    """Same nodes and state as every action; raises if it is ever executed."""

    async def refuse(state: ActionState) -> dict[str, Any]:
        raise RuntimeError(reason)

    graph: StateGraph[ActionState] = StateGraph(ActionState)
    for name in ACTION_NODES:
        graph.add_node(name, refuse)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "reason")
    graph.add_edge("reason", "writeback")
    graph.add_edge("writeback", END)
    return graph.compile(name="__schema_action__")
