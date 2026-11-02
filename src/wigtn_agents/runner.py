"""In-process runs (no server): the same assembly the worker factories build, for scripts and evals.

The runtime normally injects the checkpointer and the store; here they are in-memory and live as
long as the call. Approval points stop the run unless ``approve`` resumes it with an approve
decision for every pending request.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import RunContext
from wigtn_agents.middleware import EventSink


@dataclass
class LocalResult:
    """Outcome of one in-process run."""

    agent_id: str
    kind: str
    thread_id: str
    status: str
    answer: str = ""
    result: Optional[dict[str, Any]] = None
    pending: list[dict[str, Any]] = field(default_factory=list)

    def text(self) -> str:
        """What an eval judges: the answer, or the action result as JSON."""
        return self.answer if self.kind == "agent" else json.dumps(self.result or {}, ensure_ascii=False)


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text
    return ""


def _pending(state: Any) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for interrupt in getattr(state, "interrupts", None) or ():
        value = getattr(interrupt, "value", None) or {}
        requests.extend(value.get("action_requests", []) if isinstance(value, dict) else [])
    return requests


async def run_local(
    agent_id: str,
    payload: Any,
    ctx: RunContext,
    *,
    catalog: Catalog,
    model: Optional[BaseChatModel] = None,
    sink: Optional[EventSink] = None,
    approve: bool = False,
) -> LocalResult:
    """Run an agent (``payload`` = user text) or an action (``payload`` = input dict) once.

    Spans of the run carry the same session, user and metadata attributes as a server run, and
    the event rows carry the same ``run_id``, so a trace and its rows can be matched.
    """
    from wigtn_agents.tracing import run_attributes

    thread_id = f"local-{uuid.uuid4().hex[:12]}"
    run_id = str(uuid.uuid4())
    meta = {**trace_metadata(agent_id, ctx), "run_id": run_id}
    with run_attributes(thread_id=thread_id, principal=ctx.get("principal"), name=agent_id, metadata=meta):
        return await _run(
            agent_id, payload, ctx, thread_id, run_id, catalog=catalog, model=model, sink=sink, approve=approve
        )


def trace_metadata(agent_id: str, ctx: RunContext) -> dict[str, Any]:
    """Per-run trace metadata: which agent, batch, session type and model alias."""
    values = {
        "agent_id": agent_id,
        "batch_id": ctx.get("batch_id"),
        "session_type": ctx.get("session_type"),
        "model_alias": ctx.get("model"),
    }
    return {k: v for k, v in values.items() if isinstance(v, (str, int, float, bool))}


async def _run(
    agent_id: str,
    payload: Any,
    ctx: RunContext,
    thread_id: str,
    run_id: str,
    *,
    catalog: Catalog,
    model: Optional[BaseChatModel],
    sink: Optional[EventSink],
    approve: bool,
) -> LocalResult:
    from wigtn_agents.assembly import assemble, extra_middleware_names, resolve_run_model

    # LangChain run metadata becomes the OpenInference ``metadata`` attribute Phoenix and LangSmith read.
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "run_id": run_id},
        "metadata": {**trace_metadata(agent_id, ctx), "run_id": run_id, "thread_id": thread_id},
    }
    runtime = {"checkpointer": InMemorySaver(), "store": InMemoryStore()}

    if agent_id in catalog.action_ids():
        from wigtn_agents.actions import build_action_graph

        action = catalog.action(agent_id)
        graph: Any = build_action_graph(
            action,
            catalog,
            ctx,
            resolved=resolve_run_model(action, ctx),
            model=model,
            sink=sink,
            extra_names=extra_middleware_names(action, ctx),
        ).copy(update=runtime)
        state = await graph.ainvoke({"input": dict(payload or {})}, config=config)
        return LocalResult(agent_id, "action", thread_id, "success", result=state.get("result"))

    agent: Any = assemble(catalog.manifest(agent_id), catalog, ctx, model=model, sink=sink).copy(update=runtime)
    values = await agent.ainvoke({"messages": [{"role": "user", "content": str(payload)}]}, config=config, context=ctx)
    snapshot = await agent.aget_state(config)
    while approve and _pending(snapshot):
        decisions = [{"type": "approve"} for _ in _pending(snapshot)]
        values = await agent.ainvoke(Command(resume={"decisions": decisions}), config=config, context=ctx)
        snapshot = await agent.aget_state(config)
    pending = _pending(snapshot)
    return LocalResult(
        agent_id,
        "agent",
        thread_id,
        "interrupted" if pending else "success",
        answer=_last_ai_text(values.get("messages", [])),
        pending=pending,
    )
