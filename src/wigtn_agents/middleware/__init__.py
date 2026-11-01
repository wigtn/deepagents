"""Platform middleware. Presets that combine them live in ``wigtn_agents.presets``."""

from wigtn_agents.middleware.asset_loader import AssetLoaderMiddleware, compose_prompt, pin_assets
from wigtn_agents.middleware.events import EventRecorderMiddleware, EventSink, JsonlSink, RunStamp, build_event
from wigtn_agents.middleware.output import OutputValidatorMiddleware
from wigtn_agents.middleware.permission import DEEPAGENTS_BUILTIN_TOOLS, PermissionFilterMiddleware
from wigtn_agents.middleware.progress import ProgressMiddleware
from wigtn_agents.middleware.repeated import RepeatedToolCallMiddleware
from wigtn_agents.middleware.scope_guard import SCOPE_PARAMS, ScopeGuardMiddleware

__all__ = [
    "DEEPAGENTS_BUILTIN_TOOLS",
    "SCOPE_PARAMS",
    "AssetLoaderMiddleware",
    "EventRecorderMiddleware",
    "EventSink",
    "JsonlSink",
    "OutputValidatorMiddleware",
    "PermissionFilterMiddleware",
    "ProgressMiddleware",
    "RepeatedToolCallMiddleware",
    "RunStamp",
    "ScopeGuardMiddleware",
    "build_event",
    "compose_prompt",
    "pin_assets",
]
