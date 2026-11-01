"""Run context and thread-state extension shared by the worker factory and the middleware.

Two silent-drop boundaries live here:

* ``RunContext`` — every key a run may carry in ``config.configurable`` / ``runtime.context``.
  A key that is not declared here is discarded by LangGraph's context filter without a warning.
* ``PlatformState`` — values that must survive across runs of the same thread (checkpointed).
  Invocation-scoped values (config, context) are NOT restored on resume; state is.
"""

from __future__ import annotations

from typing import Any, NotRequired, Optional, TypedDict

from deepagents import DeepAgentState
from langgraph.graph.state import CompiledStateGraph

SESSION_TYPES: tuple[str, ...] = ("production", "evaluation", "test")

# A compiled graph of either lane; the state/context/input/output types vary per assembly.
Graph = CompiledStateGraph[Any, Any, Any, Any]


class RunContext(TypedDict, total=False):
    """Flat per-run context. Keep it flat: wrapper keys are dropped whole.

    ``model`` switches the manifest's model alias for this run; ``middleware`` adds registered
    extra middleware by name. Both select from what the image already has, they load nothing.
    """

    agent_id: str
    principal: str
    session_type: str
    batch_id: Optional[str]
    parent_run_id: Optional[str]
    entity: Optional[str]
    data_scope: list[str]
    model: Optional[str]
    middleware: list[str]


class PlatformState(DeepAgentState):
    """Thread state pinned at run start. A resumed run inherits exactly this."""

    asset_versions: NotRequired[dict[str, str]]
    run_meta: NotRequired[dict[str, Any]]


def default_context(agent_id: str, **overrides: Any) -> RunContext:
    """Build a RunContext with safe defaults for tests and scripts."""
    ctx: RunContext = {
        "agent_id": agent_id,
        "principal": "anonymous",
        "session_type": "production",
        "batch_id": None,
        "parent_run_id": None,
        "entity": None,
        "data_scope": [],
        "model": None,
        "middleware": [],
    }
    ctx.update(overrides)  # type: ignore[typeddict-item]
    return ctx
