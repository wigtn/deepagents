"""Manifest → deep agent assembly. The only place ``create_deep_agent`` is called.

Every axis is a ``build_*`` function so tests can assert the *shape* of the assembly
(middleware order, subagent suppression, allowlist) instead of trusting inline literals.
Checkpointer and store are never passed: the worker runtime injects them per run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from deepagents import SubAgent, create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import PlatformState, RunContext
from wigtn_agents.manifest import FlowTemplate, Manifest
from wigtn_agents.middleware import (
    AssetLoaderMiddleware,
    EventRecorderMiddleware,
    EventSink,
    JsonlSink,
    OutputValidatorMiddleware,
    PermissionFilterMiddleware,
    ScopeGuardMiddleware,
)
from wigtn_agents.models import build_model
from wigtn_agents.tools.gateway import ToolGatewayClient, build_tools

# First = outermost. EventRecorder sits OUTSIDE the gates so a denied call is still recorded
# (tool.allowed=false); the reference sample had it inside and never exercised the tool path.
# PermissionFilter before ScopeGuard: the gate blocks first, canonical ids are injected only
# into calls that passed.
MIDDLEWARE_ORDER: tuple[str, ...] = (
    "AssetLoaderMiddleware",
    "EventRecorderMiddleware",
    "PermissionFilterMiddleware",
    "ScopeGuardMiddleware",
    "OutputValidatorMiddleware",
)

GENERAL_PURPOSE_NAME = "general-purpose"

ExtensionFactory = Callable[..., CompiledStateGraph]


class ExtensionRegistry:
    """Code extensions baked into the image; a manifest selects one by name."""

    def __init__(self) -> None:
        self._factories: dict[str, ExtensionFactory] = {}

    def register(self, name: str, factory: ExtensionFactory) -> None:
        """Register an extension factory."""
        self._factories[name] = factory

    def build(self, name: str, **context: Any) -> CompiledStateGraph:
        """Build by name; unknown names fail — no runtime code injection."""
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"이미지에 없는 확장입니다: {name}. 런타임 코드 주입은 하지 않습니다") from exc
        return factory(**context)


def build_middleware(
    manifest: Manifest,
    catalog: Catalog,
    ctx: RunContext,
    *,
    sink: EventSink,
) -> list[AgentMiddleware]:
    """Platform middleware in the fixed order. First = outermost."""
    allowed = {catalog.resolve(ref).tool_name() for ref in manifest.permissions.allow_tools}
    return [
        AssetLoaderMiddleware(manifest, catalog),
        EventRecorderMiddleware(
            sink,
            manifest,
            session_type=ctx.get("session_type", "production"),
            batch_id=ctx.get("batch_id"),
            principal=ctx.get("principal"),
            parent_run_id=ctx.get("parent_run_id"),
            approval_tools=frozenset(build_interrupt_on(manifest, catalog) or {}),
        ),
        PermissionFilterMiddleware(allowed),
        ScopeGuardMiddleware(),
        OutputValidatorMiddleware(),
    ]


def build_subagents(manifest: Manifest) -> list[SubAgent]:
    """Suppress the auto-added general-purpose subagent (same name + ``tools: []``)."""
    constrained_gp: SubAgent = {
        "name": GENERAL_PURPOSE_NAME,
        "description": "제약된 범용 보조. 도메인 도구 없이 파일·스킬 읽기와 조사만 한다.",
        "system_prompt": (
            "너는 도메인 도구가 없는 보조 에이전트다. 파일과 지시문을 읽고 요약·조사만 한다. "
            "기록 조회나 쓰기가 필요한 요청은 범위 밖이라고 답하고 돌려보낸다."
        ),
        "tools": [],
    }
    return [constrained_gp]


SCHEMA_INTERRUPT_SENTINEL = "__schema_only_never_called__"


def build_interrupt_on(manifest: Manifest, catalog: Catalog) -> Optional[dict[str, bool]]:
    """approval-gated: interrupt before every ``approve_before`` tool.

    ``None`` (no HITL middleware, no ``after_model`` node) for agents without approval points;
    deepagents ignores an empty dict anyway. The schema-only assembly adds a sentinel entry so
    its graph is a superset of every real agent's graph (see ``worker.schema_only_graph``).
    """
    if not manifest.interrupts.approve_before:
        return None
    return {catalog.resolve(ref).tool_name(): True for ref in manifest.interrupts.approve_before}


SCHEMA_AGENT_ID = "__schema__"


def schema_manifest() -> Manifest:
    """A manifest with no assets: same state schema and node layout as any real agent.

    The worker runtime invokes the factory without run config for schema reads (assistant
    creation, thread state, Studio). State channels and node names do not depend on the manifest,
    so this assembly lets ``aget_state`` decode checkpoints written by any real agent.
    """
    return Manifest.model_validate(
        {
            "apiVersion": "wigtn.agents/v1",
            "kind": "agent",
            "metadata": {
                "name": SCHEMA_AGENT_ID,
                "version": 1,
                "owner": "platform",
                "worker_class": "interactive",
                "description": "schema-only assembly (never executed)",
            },
            "model": {"ref": "models/default"},
            "harness_preset": "schema-only",
            "skills": [],
            "prompts": {},
            "flow": {"template": "react"},
            "permissions": {"allow_tools": []},
        }
    )


def assemble(
    manifest: Manifest,
    catalog: Catalog,
    ctx: RunContext,
    *,
    model: Optional[BaseChatModel] = None,
    gateway: Optional[ToolGatewayClient] = None,
    sink: Optional[EventSink] = None,
    registry: Optional[ExtensionRegistry] = None,
    extra_tools: Sequence[BaseTool] = (),
    extra_middleware: Sequence[AgentMiddleware] = (),
    extra_interrupt_on: Optional[dict[str, bool]] = None,
) -> CompiledStateGraph:
    """Assemble the deep agent for one manifest and one run context.

    Args:
        manifest: Agent manifest (execution config, asset references only).
        catalog: Asset catalog (pins ``@active`` to integer versions).
        ctx: Flat run context (principal, session_type, scope).
        model: Chat model override (tests pass a fake). Built lazily from the manifest otherwise.
        gateway: Tool gateway client (stub by default).
        sink: Event sink (JSONL file by default).
        registry: Code-extension registry for ``flow.template == extension``.
        extra_tools: Tools added outside the catalog (e.g. ``delegate_run``).
        extra_middleware: Middleware placed OUTSIDE the platform stack (e.g. the schema-only guard).
        extra_interrupt_on: Additional HITL entries merged into the manifest's (schema-only sentinel).

    Returns:
        A compiled deep agent graph **without** checkpointer/store.

    """
    gateway = gateway or ToolGatewayClient()
    sink = sink if sink is not None else JsonlSink()
    tool_assets = catalog.tools_for_manifest(manifest)
    has_write = any(a.typed_body.side_effect == "write" for a in tool_assets)  # type: ignore[attr-defined]
    template = manifest.flow.template
    if has_write and template == FlowTemplate.REACT:
        raise ValueError("write 도구가 있는 에이전트는 react가 아니라 approval-gated여야 합니다")

    tools: list[BaseTool] = [*build_tools(manifest, catalog, gateway), *extra_tools]
    middleware = [*extra_middleware, *build_middleware(manifest, catalog, ctx, sink=sink)]
    chat_model = model or build_model(
        manifest.model.ref,
        temperature=manifest.model.temperature,
        max_tokens=manifest.model.max_tokens,
    )

    if template == FlowTemplate.EXTENSION:
        return (registry or ExtensionRegistry()).build(
            manifest.flow.extension or "",
            manifest=manifest,
            catalog=catalog,
            model=chat_model,
            tools=tools,
            middleware=middleware,
        )

    # react · plan-execute · approval-gated all become one deep agent: planning is the
    # harness's write_todos, approval is interrupt_on. plan-execute as a separate subgraph is
    # intentionally not ported (deepagents already owns the plan loop).
    interrupt_on = build_interrupt_on(manifest, catalog)
    if extra_interrupt_on:
        interrupt_on = {**(interrupt_on or {}), **extra_interrupt_on}

    return create_deep_agent(
        model=chat_model,
        tools=tools,
        middleware=middleware,
        subagents=build_subagents(manifest),
        interrupt_on=interrupt_on,
        context_schema=RunContext,
        state_schema=PlatformState,
        checkpointer=None,
        store=None,
        name=manifest.agent_id,
    )
