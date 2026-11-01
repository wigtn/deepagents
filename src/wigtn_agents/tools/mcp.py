"""MCP tool binding: one catalog tool asset ↔ one tool on an MCP server.

The asset keeps its own ``input_schema``, so the catalog (not the server) decides what the
model sees, and assembly never connects to anything. A session is opened per call; servers are
listed in ``catalog/mcp.yaml`` with ``*_env`` / ``secret://`` references only.

```yaml
servers:
  docs:
    transport: streamable_http
    url_env: WIGTN_MCP_DOCS_URL
    headers: {Authorization: "secret://mcp/docs-token"}   # sent as "Bearer <value>"
  files:
    transport: stdio
    command: uvx
    args: [some-mcp-server]
```
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field

from wigtn_agents.paths import catalog_dir
from wigtn_agents.secrets import resolve_secret

MCP_FILE = "mcp.yaml"


class McpServer(BaseModel):
    """One server entry."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio", "streamable_http", "sse"]
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    url_env: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)

    def connection(self, name: str) -> dict[str, Any]:
        """Connection dict for ``langchain_mcp_adapters``. Values are read here, at call time."""
        if self.transport == "stdio":
            if not self.command:
                raise LookupError(f"MCP 서버 {name}: stdio에는 command가 필요합니다")
            return {"transport": "stdio", "command": self.command, "args": list(self.args)}
        url = os.environ.get(self.url_env or "", "").strip()
        if not url:
            raise LookupError(f"MCP 서버 {name}: 환경 변수 {self.url_env}가 비어 있습니다")
        headers = {key: f"Bearer {resolve_secret(ref)}" for key, ref in self.headers.items()}
        return {"transport": self.transport, "url": url, "headers": headers}


class McpConfig(BaseModel):
    """``catalog/mcp.yaml``. A missing file means no servers."""

    model_config = ConfigDict(extra="forbid")

    servers: dict[str, McpServer] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> McpConfig:
        """Read the file, or return an empty config."""
        target = path or catalog_dir() / MCP_FILE
        if not target.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")) or {})


SessionFactory = Callable[[str], AbstractAsyncContextManager[Any]]


def _content_text(result: Any) -> str:
    parts = [getattr(item, "text", "") for item in getattr(result, "content", None) or []]
    return "\n".join(p for p in parts if p)


class McpToolClient:
    """Calls MCP tools. ``session_factory`` is the test seam (server name → async session)."""

    def __init__(self, config: Optional[McpConfig] = None, *, session_factory: Optional[SessionFactory] = None) -> None:
        self._config = config
        self._session_factory = session_factory
        self.calls: list[dict[str, Any]] = []

    @property
    def config(self) -> McpConfig:
        """Server list (loaded on first use)."""
        if self._config is None:
            self._config = McpConfig.load()
        return self._config

    def _session(self, server: str) -> AbstractAsyncContextManager[Any]:
        if self._session_factory is not None:
            return self._session_factory(server)
        if server not in self.config.servers:
            raise ToolException(f"catalog/{MCP_FILE}에 없는 MCP 서버입니다: {server}")
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({server: self.config.servers[server].connection(server)})  # type: ignore[dict-item]
        session: AbstractAsyncContextManager[Any] = client.session(server)
        return session

    async def acall(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Open a session, call one tool, close the session."""
        self.calls.append({"server": server, "tool": tool})
        async with self._session(server) as session:
            result = await session.call_tool(tool, {k: v for k, v in args.items() if v is not None})
        if getattr(result, "isError", False):
            raise ToolException(_content_text(result) or f"MCP 도구 오류: {server}/{tool}")
        structured = getattr(result, "structuredContent", None)
        return structured if structured is not None else _content_text(result)

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Synchronous call for sync invocations (tests, scripts); the runtime uses ``acall``."""
        return asyncio.run(self.acall(server, tool, args))
