"""HTTP tool binding: ``operation: "METHOD /path/{param}"`` on an ``api`` asset.

Base URL and credential come from ``secret://`` references at call time, so a catalog with HTTP
tools still assembles (and runs offline tests) without the variables set. Path parameters are
taken from the tool arguments; the rest go to the query string (GET, DELETE) or the JSON body.
An HTTP error becomes a tool error the model can read, not an exception that ends the run.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import quote

import httpx
from langchain_core.tools import ToolException

from wigtn_agents.assets.schema import ApiBody
from wigtn_agents.secrets import resolve_secret

_PATH_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_TIMEOUT_S = 30.0


def _request_parts(operation: str, payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    method, path = operation.split(" ", 1)
    args = dict(payload)
    for name in _PATH_PARAM.findall(path):
        if name not in args or args[name] is None:
            raise ToolException(f"경로 인자 {name}가 없습니다 ({operation})")
        if str(args[name]) in ("", ".", ".."):
            raise ToolException(f"경로 인자 {name}에 쓸 수 없는 값입니다 ({operation})")
        # Encoded as one segment: "../x" or "a/b" must not move the request to another path.
        path = path.replace("{" + name + "}", quote(str(args.pop(name)), safe=""))
    return method, path, {k: v for k, v in args.items() if v is not None}


def _headers(api: ApiBody) -> dict[str, str]:
    token = resolve_secret(api.credential_ref, required=False)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _result(response: httpx.Response) -> dict[str, Any]:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    if response.is_error:
        raise ToolException(f"HTTP {response.status_code}: {str(body)[:500]}")
    return {"status": response.status_code, "body": body}


class HttpToolClient:
    """Calls an ``api`` asset. ``transport`` is the test seam (``httpx.MockTransport``)."""

    def __init__(
        self,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._transport = transport
        self._async_transport = async_transport
        self._timeout = timeout
        self.calls: list[dict[str, Any]] = []

    def _prepare(self, api: ApiBody, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        method, path, args = _request_parts(operation, payload)
        base_url = resolve_secret(api.base_url_ref) or ""
        request: dict[str, Any] = {
            "method": method,
            "url": base_url.rstrip("/") + path,
            "headers": _headers(api),
        }
        if method in ("GET", "DELETE"):
            request["params"] = args
        else:
            request["json"] = args
        self.calls.append({"method": method, "path": path})
        return request

    def call(self, api: ApiBody, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Synchronous call."""
        request = self._prepare(api, operation, payload)
        with httpx.Client(transport=self._transport, timeout=self._timeout) as client:
            return _result(client.request(**request))

    async def acall(self, api: ApiBody, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Asynchronous call (the worker runtime runs graphs with ``ainvoke``)."""
        request = self._prepare(api, operation, payload)
        async with httpx.AsyncClient(transport=self._async_transport, timeout=self._timeout) as client:
            return _result(await client.request(**request))
