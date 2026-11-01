"""Tool gateway client (offline stand-in) and tool-asset → StructuredTool conversion.

The worker never holds connection details; a tool asset only carries a ``gateway_route``.
This stub answers canned rows so the agent loop can be exercised offline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.manifest import Manifest

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

RouteHandler = Callable[[Optional[str], dict[str, Any]], dict[str, Any]]


def _record_read(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {"record": "R-0001", "category": "A", "amount": 1_250_000, "status": "open"},
        {"record": "R-0002", "category": "A", "amount": 380_000, "status": "open"},
    ]
    return {
        "route": "record-read",
        "operation": operation,
        "period": payload.get("period"),
        "rows": rows,
        "count": len(rows),
    }


def _view_read(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    return {"route": "db-read", "view": payload.get("view"), "rows": [{"category": "B", "needs_review": True}]}


def _record_write(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    return {"route": "record-write", "operation": operation, "document_no": "R-0077"}


DEFAULT_ROUTES: dict[str, RouteHandler] = {
    "record-read": _record_read,
    "db-read": _view_read,
    "record-write": _record_write,
}


class ToolGatewayClient:
    """Routes tool calls by gateway route name. Credentials would live here, never in the worker."""

    def __init__(self, routes: Optional[dict[str, RouteHandler]] = None) -> None:
        self._routes: dict[str, RouteHandler] = dict(DEFAULT_ROUTES)
        if routes:
            self._routes.update(routes)
        self.calls: list[dict[str, Any]] = []

    def register(self, route: str, handler: RouteHandler) -> None:
        """Register or replace a route handler."""
        self._routes[route] = handler

    def call(self, route: str, operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke a gateway route. Unknown routes fail loudly."""
        self.calls.append({"route": route, "operation": operation, "payload": payload})
        try:
            handler = self._routes[route]
        except KeyError as exc:
            raise KeyError(f"게이트웨이에 없는 라우트입니다: {route}") from exc
        return handler(operation, payload)


def _args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a tool asset's JSON Schema ``input_schema`` into a Pydantic args model."""
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop, spec in schema.get("properties", {}).items():
        typ = _JSON_TYPES.get(spec.get("type", "string"), Any)
        description = spec.get("description", "")
        if prop in required:
            fields[prop] = (typ, Field(..., description=description))
        else:
            fields[prop] = (Optional[typ], Field(spec.get("default"), description=description))
    return create_model(name, **fields)


def make_tool(asset: Asset, gateway: ToolGatewayClient) -> BaseTool:
    """Wrap one tool asset as a StructuredTool that only knows its gateway route."""
    body = asset.typed_body

    def _call(**kwargs: Any) -> dict[str, Any]:
        return gateway.call(body.binding.gateway_route, body.binding.operation, kwargs)  # type: ignore[attr-defined]

    return StructuredTool.from_function(
        func=_call,
        name=asset.tool_name(),
        description=asset.description or asset.name,
        args_schema=_args_model(f"{asset.tool_name()}_args", body.input_schema),  # type: ignore[attr-defined]
    )


def build_tools(manifest: Manifest, catalog: Catalog, gateway: ToolGatewayClient) -> list[BaseTool]:
    """Skill → tool assets (pinned) → gateway-bound StructuredTools."""
    return [make_tool(asset, gateway) for asset in catalog.tools_for_manifest(manifest)]
