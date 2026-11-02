"""Aegra graph factories — the worker/assembly seam.

Registered in ``aegra.json`` as ``graphs.worker`` (agent lane) and ``graphs.action`` (action
lane). The runtime calls a factory on **every run** (no cache) with the merged run config and a
``ServerRuntime``; it loads the manifest named by ``agent_id`` and returns a freshly assembled
graph. No model, client or pool is created at import time: the runtime imports every registered
graph at startup.

Parameter detection is by type annotation, so both annotations below are load-bearing.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime

from wigtn_agents.assembly import (
    SCHEMA_AGENT_ID,
    SCHEMA_INTERRUPT_SENTINEL,
    assemble,
    extra_middleware_names,
    resolve_run_model,
    schema_manifest,
)
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import Graph, RunContext, default_context
from wigtn_agents.paths import CATALOG_DIR_ENV, catalog_dir
from wigtn_agents.tracing import ensure_server_targets

__all__ = ["CATALOG_DIR_ENV", "action", "build_run_context", "catalog_dir", "graph", "load_catalog"]
# Tracing settings are checked once when the runtime imports this module; the LangSmith OTLP
# target joins Aegra's Langfuse and Phoenix targets. Nothing is sent from here.
ensure_server_targets()

_catalog_lock = threading.Lock()
_catalog_cache: dict[str, Any] = {"stamp": None, "catalog": None}


def _dir_stamp(root: Path) -> tuple[int, float]:
    files = [p for p in root.rglob("*.yaml")]
    return len(files), max((p.stat().st_mtime for p in files), default=0.0)


def load_catalog() -> Catalog:
    """Load the YAML catalog, re-reading when any file changed (hot reload without restart)."""
    root = catalog_dir()
    stamp = _dir_stamp(root)
    with _catalog_lock:
        if _catalog_cache["stamp"] != stamp:
            _catalog_cache["catalog"] = Catalog.load(root / "assets", root / "manifests")
            _catalog_cache["stamp"] = stamp
        return _catalog_cache["catalog"]  # type: ignore[no-any-return]


def _principal_from(configurable: Mapping[str, Any], runtime: Optional[ServerRuntime[Any]]) -> str:
    """Identity comes from the auth seam only: ``langgraph_auth_user`` (server-forced)."""
    user = configurable.get("langgraph_auth_user")
    identity = None
    if isinstance(user, Mapping):
        identity = user.get("identity")
    else:
        identity = getattr(user, "identity", None)
    if not identity and runtime is not None:
        identity = getattr(getattr(runtime, "user", None), "identity", None)
    return str(identity) if identity else "anonymous"


def resolve_agent_id(configurable: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    """``agent_id`` from configurable, else context. Missing → loud failure (never a default agent)."""
    agent_id = configurable.get("agent_id") or context.get("agent_id")
    if not agent_id:
        raise LookupError(
            "agent_id가 없습니다. run 요청의 config.configurable(또는 context)에 agent_id를 포함하세요. "
            "Aegra는 assistant config와 run config를 얕게 병합하므로 configurable을 보내면 전부 보내야 합니다."
        )
    return str(agent_id)


def build_run_context(configurable: Mapping[str, Any], runtime: Optional[ServerRuntime[Any]]) -> RunContext:
    """Server-reconstructed context. Caller-supplied identity fields are ignored."""
    context: Mapping[str, Any] = {}
    if runtime is not None:
        raw = getattr(getattr(runtime, "execution_runtime", None), "context", None) or getattr(runtime, "context", None)
        if isinstance(raw, Mapping):
            context = raw
    agent_id = resolve_agent_id(configurable, context)
    session_type = configurable.get("session_type") or context.get("session_type") or "production"
    return default_context(
        agent_id,
        principal=_principal_from(configurable, runtime),
        session_type=session_type,
        batch_id=configurable.get("batch_id") or context.get("batch_id"),
        parent_run_id=configurable.get("parent_run_id") or context.get("parent_run_id"),
        entity=configurable.get("entity") or context.get("entity"),
        data_scope=list(configurable.get("data_scope") or context.get("data_scope") or []),
        model=configurable.get("model") or context.get("model"),
        middleware=list(configurable.get("middleware") or context.get("middleware") or []),
    )


def schema_only_graph(reason: str, catalog: Catalog) -> Graph:
    """Assembly returned when no ``agent_id`` is available.

    Aegra invokes the factory WITHOUT run config outside of runs: ``POST /assistants``
    (``get_graph_for_validation``), thread state/history reads (``access_context="threads.read"``,
    config carries only ``thread_id``), Studio topology. Those paths need a graph whose state
    schema and node names match the real agents so checkpoints decode; they never execute it.
    A guard middleware makes execution without ``agent_id`` fail loudly anyway.
    """
    from wigtn_agents.middleware.extra import MiddlewareSpec, available_middleware, build_extra_middleware
    from wigtn_agents.middleware.guard import MissingAgentGuardMiddleware
    from wigtn_agents.testing import ScriptedFakeModel

    manifest = schema_manifest()
    # An identified principal, so memory mounts and its node and channel are in the reader too.
    ctx = default_context(SCHEMA_AGENT_ID, principal=SCHEMA_AGENT_ID)
    # Every registered extra middleware, in case one adds a node (see middleware.extra).
    extras = build_extra_middleware(available_middleware(), MiddlewareSpec(manifest, ctx, schema_only=True))
    return assemble(
        manifest,
        catalog,
        ctx,
        model=ScriptedFakeModel(),  # never called: the guard raises first
        sink=[],
        outer_middleware=[MissingAgentGuardMiddleware(reason)],
        inner_middleware=extras,
        # HITL middleware adds a node + channel; include it so this graph is a SUPERSET of every
        # real assembly (approval-gated threads keep their pending interrupt tasks when read).
        extra_interrupt_on={SCHEMA_INTERRUPT_SENTINEL: True},
    )


def graph(config: RunnableConfig, runtime: ServerRuntime[RunContext]) -> Graph:
    """Per-request factory: manifest → deep agent. Checkpointer/store are injected by Aegra."""
    configurable: Mapping[str, Any] = config.get("configurable", {}) if config else {}
    catalog = load_catalog()
    try:
        ctx = build_run_context(configurable, runtime)
    except LookupError as exc:
        return schema_only_graph(str(exc), catalog)
    return assemble(catalog.manifest(ctx["agent_id"]), catalog, ctx)


def action(config: RunnableConfig, runtime: ServerRuntime[RunContext]) -> Graph:
    """Per-request factory of the action lane: action manifest → gather/reason/writeback graph."""
    from wigtn_agents.actions import build_action_graph, schema_only_action_graph

    configurable: Mapping[str, Any] = config.get("configurable", {}) if config else {}
    catalog = load_catalog()
    try:
        ctx = build_run_context(configurable, runtime)
    except LookupError as exc:
        return schema_only_action_graph(str(exc))
    manifest = catalog.action(ctx["agent_id"])
    return build_action_graph(
        manifest,
        catalog,
        ctx,
        resolved=resolve_run_model(manifest, ctx),
        extra_names=extra_middleware_names(manifest, ctx),
    )
