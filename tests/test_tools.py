"""Tool bindings: gateway (stub), HTTP on an api asset, MCP server tools."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.secrets import resolve_secret, secret_env_name
from wigtn_agents.tools.gateway import ToolClients, make_tool
from wigtn_agents.tools.http import HttpToolClient
from wigtn_agents.tools.mcp import McpConfig, McpServer, McpToolClient


def _tool(name: str, binding: dict[str, Any], *, side_effect: str = "read") -> Asset:
    return Asset.model_validate(
        {
            "kind": "tool",
            "name": name,
            "version": 1,
            "status": "active",
            "owner": "t",
            "description": f"{name} tool",
            "body": {
                "input_schema": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}, "period": {"type": "string"}},
                    "required": ["record_id"],
                },
                "side_effect": side_effect,
                "idempotent": True,
                "binding": binding,
            },
        }
    )


def test_secret_참조는_환경_변수_이름으로_바뀐다(monkeypatch: pytest.MonkeyPatch) -> None:
    assert secret_env_name("secret://records/base-url") == "WIGTN_SECRET_RECORDS_BASE_URL"
    monkeypatch.delenv("WIGTN_SECRET_RECORDS_BASE_URL", raising=False)
    with pytest.raises(LookupError, match="WIGTN_SECRET_RECORDS_BASE_URL"):
        resolve_secret("secret://records/base-url")
    assert resolve_secret("secret://records/base-url", required=False) is None


def test_바인딩_형식이_틀리면_자산이_로드되지_않는다() -> None:
    with pytest.raises(ValueError, match="METHOD"):
        _tool("bad", {"type": "http", "api": "api://records-api@1", "operation": "FETCH records"})
    with pytest.raises(ValueError, match="server"):
        _tool("bad", {"type": "mcp", "tool": "search"})
    with pytest.raises(ValueError, match="gateway_route"):
        _tool("bad", {"type": "gateway"})


def test_http_도구는_api_자산의_주소와_자격으로_호출한다(catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIGTN_SECRET_RECORDS_BASE_URL", "https://records.example/api/")
    monkeypatch.setenv("WIGTN_SECRET_RECORDS_SERVICE_ACCOUNT", "tok-1")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"record": "R-1", "status": "open"})

    clients = ToolClients(http=HttpToolClient(transport=httpx.MockTransport(handler)))
    asset = _tool("record-get", {"type": "http", "api": "api://records-api@1", "operation": "GET /records/{record_id}"})
    result = make_tool(asset, clients, catalog).invoke({"record_id": "R-1", "period": "2026-08"})

    assert result == {"status": 200, "body": {"record": "R-1", "status": "open"}}
    assert str(seen[0].url) == "https://records.example/api/records/R-1?period=2026-08"
    assert seen[0].headers["Authorization"] == "Bearer tok-1"


def test_http_오류는_모델이_읽는_도구_오류가_된다(catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIGTN_SECRET_RECORDS_BASE_URL", "https://records.example")
    clients = ToolClients(http=HttpToolClient(transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no"))))
    asset = _tool("record-post", {"type": "http", "api": "api://records-api@1", "operation": "POST /records"})
    out = make_tool(asset, clients, catalog).invoke(
        {"type": "tool_call", "name": "record-post", "id": "c1", "args": {"record_id": "R-9"}}
    )
    assert out.status == "error" and "HTTP 404" in out.text


async def test_mcp_도구는_서버의_원격_도구를_부른다() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Session:
        async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
            calls.append((name, args))
            return SimpleNamespace(isError=False, structuredContent=None, content=[SimpleNamespace(text="found 2")])

    @asynccontextmanager
    async def session(server: str) -> AsyncIterator[Session]:
        assert server == "docs"
        yield Session()

    clients = ToolClients(mcp=McpToolClient(session_factory=session))
    asset = _tool("docs-search", {"type": "mcp", "server": "docs", "tool": "search"})
    result = await make_tool(asset, clients).ainvoke({"record_id": "R-1"})
    assert result == "found 2"
    assert calls == [("search", {"record_id": "R-1"})]


def test_mcp_서버_설정은_값을_실행_시점에_읽는다(monkeypatch: pytest.MonkeyPatch) -> None:
    server = McpServer(
        transport="streamable_http", url_env="WIGTN_MCP_DOCS_URL", headers={"Authorization": "secret://mcp/docs"}
    )
    monkeypatch.delenv("WIGTN_MCP_DOCS_URL", raising=False)
    with pytest.raises(LookupError, match="WIGTN_MCP_DOCS_URL"):
        server.connection("docs")
    monkeypatch.setenv("WIGTN_MCP_DOCS_URL", "http://localhost:9000/mcp")
    monkeypatch.setenv("WIGTN_SECRET_MCP_DOCS", "t")
    assert server.connection("docs") == {
        "transport": "streamable_http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer t"},
    }


def test_mcp_설정_파일이_없으면_서버가_없다(tmp_path: Path) -> None:
    assert McpConfig.load(tmp_path / "missing.yaml").servers == {}
    path = tmp_path / "mcp.yaml"
    path.write_text("servers:\n  files: {transport: stdio, command: uvx, args: [some-server]}\n")
    assert McpConfig.load(path).servers["files"].connection("files")["command"] == "uvx"


def test_gateway_도구는_기존_스텁으로_간다(catalog: Catalog) -> None:
    clients = ToolClients()
    tool = make_tool(catalog.resolve("tool://record-read@active"), clients, catalog)
    out = tool.invoke({"period": "2026-08", "entity": "1000"})
    assert out["count"] == 2 and clients.gateway.calls[0]["route"] == "record-read"
    assert json.dumps(out)
