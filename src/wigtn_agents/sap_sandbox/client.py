"""Read-only client for the SAP Business Accelerator Hub sandbox (S/4HANA Cloud OData v2).

The sandbox (``https://sandbox.api.sap.com/s4hanacloud``) answers GET requests with a personal
API key in the ``APIKey`` header; it has no write access. The key comes from api.sap.com (log in
with a SAP ID, open an API, "Show API Key") and lives in ``SAP_SANDBOX_API_KEY``.

Sandbox data does not change, so answers are cached per process by URL: repeated runs of an eval
do not spend the sandbox's request quota.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import UTC, datetime
from typing import Any, Optional

import httpx

API_KEY_ENV = "SAP_SANDBOX_API_KEY"
BASE_URL_ENV = "SAP_SANDBOX_BASE_URL"
DEFAULT_BASE_URL = "https://sandbox.api.sap.com/s4hanacloud"
_DATE = re.compile(r"/Date\((-?\d+)(?:[+-]\d+)?\)/")


class SapError(RuntimeError):
    """The sandbox refused or failed the request (missing key, quota, unknown service)."""


def odata_value(value: Any) -> Any:
    """``/Date(ms)/`` → ISO date; everything else unchanged."""
    if isinstance(value, str):
        match = _DATE.fullmatch(value)
        if match:
            return datetime.fromtimestamp(int(match.group(1)) / 1000, tz=UTC).date().isoformat()
    return value


def quote(value: str) -> str:
    """An OData string literal."""
    return "'" + value.replace("'", "''") + "'"


class SapSandboxClient:
    """GET entity sets of the sandbox's OData services."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self._transport = transport
        self._timeout = timeout
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.requests = 0

    def _key(self) -> str:
        key = (self._api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise SapError(f"{API_KEY_ENV}가 비어 있습니다. api.sap.com에서 SAP ID로 로그인해 API 키를 받아 넣으세요")
        return key

    def get(
        self,
        service: str,
        entity: str,
        *,
        filters: Optional[list[str]] = None,
        select: Optional[list[str]] = None,
        orderby: Optional[str] = None,
        top: int = 100,
    ) -> list[dict[str, Any]]:
        """Rows of ``/sap/opu/odata/sap/<service>/<entity>`` (``d.results``), dates as ISO strings."""
        params: dict[str, str] = {"$format": "json", "$top": str(top)}
        if filters:
            params["$filter"] = " and ".join(filters)
        if select:
            params["$select"] = ",".join(select)
        if orderby:
            params["$orderby"] = orderby
        url = f"{self._base_url}/sap/opu/odata/sap/{service}/{entity}"
        cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        headers = {"APIKey": self._key(), "Accept": "application/json"}
        with httpx.Client(transport=self._transport, timeout=self._timeout) as http:
            response = http.get(url, params=params, headers=headers)
        self.requests += 1
        if response.status_code != 200:
            raise SapError(f"SAP 샌드박스 {service}/{entity}: HTTP {response.status_code} {response.text[:200]}")
        body = response.json().get("d", {})
        rows = body.get("results", body if isinstance(body, list) else [])
        clean = [{k: odata_value(v) for k, v in row.items() if k != "__metadata"} for row in rows]
        with self._lock:
            self._cache[cache_key] = clean
        return clean
