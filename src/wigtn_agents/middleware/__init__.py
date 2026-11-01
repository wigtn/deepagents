"""Platform middleware: asset loader, permission filter, scope guard, event recorder, output validator."""

from wigtn_agents.middleware.asset_loader import AssetLoaderMiddleware
from wigtn_agents.middleware.events import EventRecorderMiddleware, EventSink, JsonlSink
from wigtn_agents.middleware.output import OutputValidatorMiddleware
from wigtn_agents.middleware.permission import DEEPAGENTS_BUILTIN_TOOLS, PermissionFilterMiddleware
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
    "ScopeGuardMiddleware",
]
