"""Middleware presets: three fixed stacks, chosen per manifest with ``harness_preset``.

| preset     | stack (first = outermost)                                                        |
|------------|----------------------------------------------------------------------------------|
| `observe`  | AssetLoader → EventRecorder → Progress → [extra]                                 |
| `standard` | observe + PermissionFilter → ScopeGuard → RepeatedToolCall → OutputValidator     |
| `strict`   | standard + ModelCallLimit → ToolCallLimit, and no filesystem writes              |

``observe`` measures the bare harness: no gate between the model and its tools, so it is for
baselines on test data, not for real scopes. ``standard`` is the default. ``strict`` adds hard
run limits and makes the built-in filesystem read-only.

The recorder is outside every gate, so a denied call is still a ``tool.call`` row
(``allowed: false``). Extra middleware sits right inside the recorder and outside the gates.
The action lane uses the same presets for its tool loop, without the asset loader (its prompt
is composed once) and without the output validator (its output is a schema).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Optional

from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware

from wigtn_agents.middleware import (
    EventRecorderMiddleware,
    OutputValidatorMiddleware,
    PermissionFilterMiddleware,
    ProgressMiddleware,
    RepeatedToolCallMiddleware,
    ScopeGuardMiddleware,
)

STRICT_MODEL_CALL_LIMIT: Final = 40
STRICT_TOOL_CALL_LIMIT: Final = 60

PRESET_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "observe": ("AssetLoaderMiddleware", "EventRecorderMiddleware", "ProgressMiddleware"),
    "standard": (
        "AssetLoaderMiddleware",
        "EventRecorderMiddleware",
        "ProgressMiddleware",
        "PermissionFilterMiddleware",
        "ScopeGuardMiddleware",
        "RepeatedToolCallMiddleware",
        "OutputValidatorMiddleware",
    ),
    "strict": (
        "AssetLoaderMiddleware",
        "EventRecorderMiddleware",
        "ProgressMiddleware",
        "PermissionFilterMiddleware",
        "ScopeGuardMiddleware",
        "RepeatedToolCallMiddleware",
        "OutputValidatorMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ),
}


def permission_paths(root: str) -> list[str]:
    """``root`` and everything under it, hidden entries included (``/**`` skips dot paths)."""
    base = root.rstrip("/")
    return [base or "/", f"{base}/**", f"{base}/**/.*", f"{base}/**/.*/**"]


def preset_permissions(preset: str) -> Optional[list[FilesystemPermission]]:
    """Filesystem permissions of a preset (``strict``: no writes anywhere)."""
    if preset == "strict":
        return [FilesystemPermission(operations=["write"], paths=permission_paths("/"), mode="deny")]
    return None


PRESET_RANK: Final[dict[str, int]] = {"observe": 0, "standard": 1, "strict": 2}


def stricter(a: str, b: str) -> str:
    """The stricter of two presets (a subagent never runs with fewer gates than its parent)."""
    return a if PRESET_RANK[a] >= PRESET_RANK[b] else b


def limit_middleware(model_calls: int = STRICT_MODEL_CALL_LIMIT, *, tools: bool = True) -> list[AgentMiddleware]:
    """Run limits that end the run cleanly instead of raising."""
    limits: list[Any] = [ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end")]
    if tools:
        limits.append(ToolCallLimitMiddleware(run_limit=STRICT_TOOL_CALL_LIMIT, exit_behavior="continue"))
    return limits


def loop_stack(
    preset: str,
    *,
    recorder: EventRecorderMiddleware,
    allowed_tools: set[str],
    extra: Sequence[AgentMiddleware] = (),
    head: Sequence[AgentMiddleware] = (),
    static_context: Optional[Mapping[str, Any]] = None,
    output_validator: bool = True,
    model_call_limit: Optional[int] = None,
) -> list[AgentMiddleware]:
    """Stack for one agent loop in the decided order. ``head`` goes first (the asset loader).

    ``model_call_limit`` bounds the loop in any preset (the action lane's rounds); strict takes
    the smaller of it and its own limit.
    """
    if preset not in PRESET_ORDER:
        raise ValueError(f"알 수 없는 프리셋입니다: {preset} (observe · standard · strict)")
    stack: list[AgentMiddleware] = [*head, recorder, ProgressMiddleware(), *extra]
    if preset == "observe":
        if model_call_limit is not None:
            stack += limit_middleware(model_call_limit, tools=False)
        return stack
    stack += [
        PermissionFilterMiddleware(allowed_tools),
        ScopeGuardMiddleware(static_context=static_context),
        RepeatedToolCallMiddleware(),
    ]
    if output_validator:
        stack.append(OutputValidatorMiddleware())
    if preset == "strict":
        stack += limit_middleware(min(model_call_limit or STRICT_MODEL_CALL_LIMIT, STRICT_MODEL_CALL_LIMIT))
    elif model_call_limit is not None:
        stack += limit_middleware(model_call_limit, tools=False)
    return stack
