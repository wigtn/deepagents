"""Tool-asset → StructuredTool conversion and the offline gateway stand-in.

The worker never holds connection details. A tool asset carries a binding: a gateway route,
an HTTP operation on an ``api`` asset, or a tool on an MCP server.

Gateway routes answered here: the sample ERP (``wigtn_agents.sample_erp``, the shipped catalog's
tools), ``delegate-run`` (records the hand-off, the router's tool), and canned rows for the small
"records" catalog the runtime tests use (``tests/fixtures/catalog``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import ApiBody, Asset, ToolBody
from wigtn_agents.manifest import AnyManifest
from wigtn_agents.tools.http import HttpToolClient
from wigtn_agents.tools.mcp import McpToolClient

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

RouteHandler = Callable[[Optional[str], dict[str, Any]], dict[str, Any]]


# -- tests/fixtures/catalog ("records" example domain) ----------------------------------------


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


def _delegate_run(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    """The router's hand-off. Starting the other agent's run is the caller's job; this records the choice."""
    return {"route": "delegate-run", "agent_id": payload.get("agent_id"), "status": "accepted"}


FIXTURE_ROUTES: dict[str, RouteHandler] = {
    "record-read": _record_read,
    "db-read": _view_read,
    "record-write": _record_write,
}


def default_routes() -> dict[str, RouteHandler]:
    """Every route a new gateway client answers (the sample ERP is shared by the process)."""
    from wigtn_agents.sample_erp import erp_routes

    return {**FIXTURE_ROUTES, "delegate-run": _delegate_run, **erp_routes()}


class ToolGatewayClient:
    """Routes tool calls by gateway route name. Credentials would live here, never in the worker."""

    def __init__(self, routes: Optional[dict[str, RouteHandler]] = None) -> None:
        self._routes: dict[str, RouteHandler] = default_routes()
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


def schema_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a flat JSON Schema (tool ``input_schema``, action ``output_schema``) into a Pydantic model."""
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


class ToolClients:
    """Everything a tool call can go through. Defaults: offline gateway stub, real HTTP, real MCP."""

    def __init__(
        self,
        gateway: Optional[ToolGatewayClient] = None,
        http: Optional[HttpToolClient] = None,
        mcp: Optional[McpToolClient] = None,
    ) -> None:
        self.gateway = gateway or ToolGatewayClient()
        self.http = http or HttpToolClient()
        self.mcp = mcp or McpToolClient()


def make_tool(asset: Asset, clients: ToolClients, catalog: Optional[Catalog] = None) -> BaseTool:
    """Wrap one tool asset as a StructuredTool bound to its gateway route, HTTP operation or MCP tool."""
    body: ToolBody = asset.typed_body  # type: ignore[assignment]
    binding = body.binding
    args_schema = schema_model(f"{asset.tool_name()}_args", body.input_schema)
    common: dict[str, Any] = {
        "name": asset.tool_name(),
        "description": asset.description or asset.name,
        "args_schema": args_schema,
        "handle_tool_error": True,
    }

    if binding.type == "gateway":

        def _gateway(**kwargs: Any) -> dict[str, Any]:
            return clients.gateway.call(binding.gateway_route or "", binding.operation, kwargs)

        return StructuredTool.from_function(func=_gateway, **common)

    if binding.type == "http":
        if catalog is None:
            raise ValueError(f"{asset.ref}: http 바인딩은 api 자산을 해석할 카탈로그가 필요합니다")
        api: ApiBody = catalog.resolve(binding.api or "").typed_body  # type: ignore[assignment]
        operation = binding.operation or ""

        def _http(**kwargs: Any) -> dict[str, Any]:
            return clients.http.call(api, operation, kwargs)

        async def _ahttp(**kwargs: Any) -> dict[str, Any]:
            return await clients.http.acall(api, operation, kwargs)

        return StructuredTool.from_function(func=_http, coroutine=_ahttp, **common)

    server, remote = binding.server or "", binding.tool or ""

    def _mcp(**kwargs: Any) -> Any:
        return clients.mcp.call(server, remote, kwargs)

    async def _amcp(**kwargs: Any) -> Any:
        return await clients.mcp.acall(server, remote, kwargs)

    return StructuredTool.from_function(func=_mcp, coroutine=_amcp, **common)


def build_tools(manifest: AnyManifest, catalog: Catalog, clients: ToolClients) -> list[BaseTool]:
    """Skill → tool assets (pinned) → bound StructuredTools."""
    return [make_tool(asset, clients, catalog) for asset in catalog.tools_for_manifest(manifest)]
