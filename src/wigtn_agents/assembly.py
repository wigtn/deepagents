"""Manifest → deep agent assembly. The only place a chat-lane ``create_deep_agent`` is called.

Every axis is a ``build_*`` function so tests can assert the *shape* of the assembly
(middleware order, subagent suppression, allowlist) instead of trusting inline literals.
Checkpointer and store are never passed: the worker runtime injects them per run.

Per run, three things can differ from the manifest without editing it, each selecting only from
what the image already has: the model alias (``ctx.model``), extra middleware names
(``ctx.middleware``) and the run context itself (principal, scope, batch).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Optional, cast

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import Graph, PlatformState, RunContext
from wigtn_agents.manifest import AnyManifest, FlowTemplate, Manifest
from wigtn_agents.middleware import (
    AssetLoaderMiddleware,
    EventRecorderMiddleware,
    EventSink,
    JsonlSink,
    RunStamp,
    compose_prompt,
    pin_assets,
)
from wigtn_agents.middleware.extra import MiddlewareSpec, build_extra_middleware, merge_names
from wigtn_agents.middleware.progress import ProgressMiddleware
from wigtn_agents.models import ModelRegistry, ResolvedModel, build_model, fake_forced, load_registry
from wigtn_agents.presets import PRESET_ORDER, loop_stack, permission_paths, preset_permissions, stricter
from wigtn_agents.tools.gateway import ToolClients, ToolGatewayClient, build_tools

# First = outermost, per preset (see wigtn_agents.presets). Kept here for the shape tests.
MIDDLEWARE_ORDER: dict[str, tuple[str, ...]] = PRESET_ORDER

GENERAL_PURPOSE_NAME = "general-purpose"
MEMORY_ROUTE = "/memories/"
MEMORY_FILE = "/memories/AGENTS.md"

ExtensionFactory = Callable[..., Graph]


class ExtensionRegistry:
    """Code extensions baked into the image; a manifest selects one by name."""

    def __init__(self) -> None:
        self._factories: dict[str, ExtensionFactory] = {}

    def register(self, name: str, factory: ExtensionFactory) -> None:
        """Register an extension factory."""
        self._factories[name] = factory

    def build(self, name: str, **context: Any) -> Graph:
        """Build by name; unknown names fail — no runtime code injection."""
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"이미지에 없는 확장입니다: {name}. 런타임 코드 주입은 하지 않습니다") from exc
        return factory(**context)


def resolve_run_model(
    manifest: AnyManifest, ctx: RunContext, registry: Optional[ModelRegistry] = None
) -> ResolvedModel:
    """The run's model alias wins over the manifest's; both must be registered.

    The scripted fake is refused for production runs unless the whole process runs fake
    (``WIGTN_AGENT_FAKE_MODEL=1``): a fake result must never be written back as a real one.
    """
    resolved = (registry or load_registry()).resolve(ctx.get("model") or manifest.model.ref)
    if resolved.provider == "fake" and ctx.get("session_type", "production") == "production" and not fake_forced():
        raise ValueError(f"{resolved.alias}: production run에는 fake 모델을 쓸 수 없습니다 (session_type을 test로)")
    return resolved


def extra_middleware_names(manifest: AnyManifest, ctx: RunContext) -> tuple[str, ...]:
    """Manifest names first, then the run's, without duplicates."""
    return merge_names(manifest.extra_middleware, ctx.get("middleware") or [])


def allowed_tool_names(manifest: Manifest, catalog: Catalog) -> set[str]:
    """LLM-facing names of the manifest's ``allow_tools``."""
    return {catalog.resolve(ref).tool_name() for ref in manifest.permissions.allow_tools}


def build_middleware(
    manifest: Manifest,
    catalog: Catalog,
    ctx: RunContext,
    *,
    sink: EventSink,
    model: ResolvedModel,
    extra: Sequence[AgentMiddleware] = (),
    extra_names: tuple[str, ...] = (),
) -> list[AgentMiddleware]:
    """The manifest's preset stack. First = outermost."""
    recorder = EventRecorderMiddleware(
        sink,
        RunStamp.for_manifest(manifest, ctx, model, extra_middleware=extra_names),
        approval_tools=frozenset(build_interrupt_on(manifest, catalog) or {}),
    )
    return loop_stack(
        manifest.harness_preset,
        recorder=recorder,
        allowed_tools=allowed_tool_names(manifest, catalog),
        extra=extra,
        head=[AssetLoaderMiddleware(manifest, catalog, model=model)],
        static_context=dict(ctx),
    )


def subagent_permissions(preset: str) -> list[FilesystemPermission]:
    """Subagents never write ``/memories/``; strict ones write nothing (these replace the parent's)."""
    return [
        *(preset_permissions(preset) or []),
        FilesystemPermission(operations=["write"], paths=permission_paths(MEMORY_ROUTE), mode="deny"),
    ]


def constrained_general_purpose(
    *, recorder: Optional[EventRecorderMiddleware] = None, preset: str = "standard"
) -> SubAgent:
    """Replaces deepagents' auto-added general-purpose subagent (same name, no tools).

    The auto-added one inherits every parent tool and runs them outside the parent's middleware,
    so ``task`` would be a way around the permission gate. With a recorder its model and file
    calls are in the run record like everyone else's.
    """
    spec: SubAgent = {
        "name": GENERAL_PURPOSE_NAME,
        "description": "제약된 범용 보조. 도메인 도구 없이 파일·스킬 읽기와 조사만 한다.",
        "system_prompt": (
            "너는 도메인 도구가 없는 보조 에이전트다. 파일과 지시문을 읽고 요약·조사만 한다. "
            "기록 조회나 쓰기가 필요한 요청은 범위 밖이라고 답하고 돌려보낸다."
        ),
        "tools": [],
        "permissions": subagent_permissions(preset),
    }
    if recorder is not None:
        spec["middleware"] = [recorder, ProgressMiddleware()]
    return spec


def subagent_stamp(stamp: RunStamp, agent_id: str) -> RunStamp:
    """The parent's stamp, re-labelled for a subagent without a manifest."""
    return replace(stamp, agent_id=agent_id, role="subagent", skill=None)


def build_subagents(
    manifest: Manifest,
    catalog: Optional[Catalog] = None,
    ctx: Optional[RunContext] = None,
    *,
    clients: Optional[ToolClients] = None,
    sink: Optional[EventSink] = None,
    model: Optional[ResolvedModel] = None,
    parent_stamp: Optional[RunStamp] = None,
) -> list[SubAgent]:
    """Manifest subagents, then the constrained general-purpose one (always last).

    A subagent is another catalog agent: its prompts and skills become its system prompt and its
    tools are its own. Subagent tool calls never pass through the parent's middleware, so each
    subagent gets a full stack: the stricter of its own and the parent's preset, the extra
    middleware of the parent manifest, its own manifest and the run, and the run context passed
    in statically (a subagent graph does not receive the parent's runtime context). Subagents
    inherit the parent's model (one backbone per run), must be read-only (approval inside a
    subagent is not supported), never write ``/memories/`` and are one level deep.
    """
    specs: list[SubAgent] = []
    if manifest.subagents:
        if catalog is None or ctx is None or model is None:
            raise ValueError("서브에이전트 조립에는 catalog · ctx · model이 필요합니다")
        clients = clients or ToolClients()
        sink = sink if sink is not None else JsonlSink()
        for name in manifest.subagents:
            sub = catalog.manifest(name)
            if sub.subagents:
                raise ValueError(f"{name}: 서브에이전트는 한 단계만 둡니다")
            writes = [a.ref for a in catalog.tools_for_manifest(sub) if a.typed_body.side_effect == "write"]  # type: ignore[attr-defined]
            if writes or sub.interrupts.approve_before:
                raise ValueError(f"{name}: 서브에이전트는 읽기 전용이어야 합니다 ({writes})")
            pinned = pin_assets(sub, catalog)
            preset = stricter(manifest.harness_preset, sub.harness_preset)
            names = merge_names(manifest.extra_middleware, sub.extra_middleware, ctx.get("middleware") or [])
            stamp = replace(
                RunStamp.for_manifest(sub, ctx, model, extra_middleware=names, role="subagent"), preset=preset
            )
            recorder = EventRecorderMiddleware(sink, stamp, asset_versions=pinned)
            spec: SubAgent = {
                "name": name,
                "description": sub.metadata.description or name,
                "system_prompt": compose_prompt(sub, catalog, pinned),
                "tools": build_tools(sub, catalog, clients),
                "middleware": loop_stack(
                    preset,
                    recorder=recorder,
                    allowed_tools=allowed_tool_names(sub, catalog),
                    extra=build_extra_middleware(names, MiddlewareSpec(sub, ctx)),
                    static_context=dict(ctx),
                ),
                "permissions": subagent_permissions(preset),
            }
            specs.append(spec)
    gp_recorder = None
    if parent_stamp is not None and sink is not None:
        gp_recorder = EventRecorderMiddleware(
            sink, subagent_stamp(parent_stamp, GENERAL_PURPOSE_NAME), asset_versions={}
        )
    specs.append(constrained_general_purpose(recorder=gp_recorder, preset=manifest.harness_preset))
    return specs


_fallback_store: dict[str, Any] = {}
logger = logging.getLogger(__name__)


class RuntimeStoreBackend(StoreBackend):
    """``StoreBackend`` on the runtime-injected store, with a process-local fallback.

    Aegra injects a Postgres store per run. A run without one (a local script, a test) gets an
    in-memory store instead of failing on the first memory read; it lives as long as the process,
    and the first fallback is logged so it is never mistaken for persistence.
    """

    def _get_store(self) -> Any:
        try:
            store = super()._get_store()
        except RuntimeError:
            store = None
        if store is not None:
            return store
        if "store" not in _fallback_store:
            from langgraph.store.memory import InMemoryStore

            logger.warning("no runtime store; /memories/ uses a process-local in-memory store")
            _fallback_store["store"] = InMemoryStore()
        return _fallback_store["store"]


def memory_namespace(manifest: AnyManifest, ctx: RunContext) -> tuple[str, ...]:
    """Store namespace of ``/memories/``: one per (agent, principal), shared across threads."""
    return ("memories", manifest.agent_id, str(ctx.get("principal") or "anonymous"))


def build_backend(manifest: Manifest, ctx: RunContext) -> Optional[BackendProtocol]:
    """``memory: true`` mounts ``/memories/`` on the runtime store; everything else is thread scratch.

    Memory needs an identified principal. Without one (``AUTH_TYPE=noop`` makes every caller
    ``anonymous``) a single namespace would be shared by every caller and one caller's notes
    would reach everyone's prompt, so memory stays off for that run.
    """
    if not manifest.memory:
        return None
    if str(ctx.get("principal") or "anonymous") == "anonymous":
        logger.info("memory off for %s: no identified principal", manifest.agent_id)
        return None
    namespace = memory_namespace(manifest, ctx)
    return CompositeBackend(
        default=StateBackend(),
        routes={MEMORY_ROUTE: RuntimeStoreBackend(namespace=lambda runtime: namespace)},
    )


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
            # strict ⊇ standard ⊇ observe node-wise, and memory adds a node and a channel:
            # the reader must be a superset of every real agent.
            "harness_preset": "strict",
            "memory": True,
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
    model_registry: Optional[ModelRegistry] = None,
    clients: Optional[ToolClients] = None,
    gateway: Optional[ToolGatewayClient] = None,
    sink: Optional[EventSink] = None,
    registry: Optional[ExtensionRegistry] = None,
    extra_tools: Sequence[BaseTool] = (),
    outer_middleware: Sequence[AgentMiddleware] = (),
    inner_middleware: Sequence[AgentMiddleware] = (),
    extra_interrupt_on: Optional[dict[str, bool]] = None,
) -> Graph:
    """Assemble the deep agent for one manifest and one run context.

    Args:
        manifest: Agent manifest (execution config, asset references only).
        catalog: Asset catalog (pins ``@active`` to integer versions).
        ctx: Flat run context (principal, session_type, scope, model, middleware).
        model: Chat model override (tests pass a fake). Built lazily from the registry otherwise.
        model_registry: Model registry override (tests); ``catalog/models.yaml`` otherwise.
        clients: Tool clients (gateway stub, HTTP, MCP).
        gateway: Gateway client shortcut when ``clients`` is not given.
        sink: Event sink (JSONL file by default).
        registry: Code-extension registry for ``flow.template == extension``.
        extra_tools: Tools added outside the catalog (e.g. ``delegate_run``).
        outer_middleware: Middleware OUTSIDE the platform stack (e.g. the schema-only guard).
        inner_middleware: Middleware instances placed in the extra slot after the registered names.
        extra_interrupt_on: Additional HITL entries merged into the manifest's (schema-only sentinel).

    Returns:
        A compiled deep agent graph **without** checkpointer/store.

    """
    clients = clients or ToolClients(gateway=gateway)
    sink = sink if sink is not None else JsonlSink()
    resolved = resolve_run_model(manifest, ctx, model_registry)
    tool_assets = catalog.tools_for_manifest(manifest)
    has_write = any(a.typed_body.side_effect == "write" for a in tool_assets)  # type: ignore[attr-defined]
    template = manifest.flow.template
    if has_write and template == FlowTemplate.REACT:
        raise ValueError("write 도구가 있는 에이전트는 react가 아니라 approval-gated여야 합니다")

    names = extra_middleware_names(manifest, ctx)
    extra = [*build_extra_middleware(names, MiddlewareSpec(manifest, ctx)), *inner_middleware]
    tools: list[BaseTool] = [*build_tools(manifest, catalog, clients), *extra_tools]
    platform = build_middleware(manifest, catalog, ctx, sink=sink, model=resolved, extra=extra, extra_names=names)
    middleware = [*outer_middleware, *platform]
    stamp = next(m.stamp for m in platform if isinstance(m, EventRecorderMiddleware))
    chat_model = model or build_model(resolved.alias, registry=model_registry, **manifest.model.params())

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

    options: dict[str, Any] = {}
    backend = build_backend(manifest, ctx)
    if backend is not None:
        options["backend"] = backend
        options["memory"] = [MEMORY_FILE]
    permissions = preset_permissions(manifest.harness_preset)
    if permissions:
        options["permissions"] = permissions

    return create_deep_agent(
        model=chat_model,
        tools=tools,
        middleware=middleware,
        subagents=build_subagents(
            manifest, catalog, ctx, clients=clients, sink=sink, model=resolved, parent_stamp=stamp
        ),
        interrupt_on=cast(Optional[dict[str, Any]], interrupt_on),
        context_schema=RunContext,
        state_schema=PlatformState,
        checkpointer=None,
        store=None,
        name=manifest.agent_id,
        **options,
    )
