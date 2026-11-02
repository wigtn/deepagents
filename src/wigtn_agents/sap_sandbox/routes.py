"""Gateway routes over SAP FI on the sandbox: one route per ``catalog/assets/tool.sap-fi-*.yaml``.

``company`` comes from the run context through the scope guard. A refusal (FI rule, sandbox
error, missing API key) raises ``ToolException``, so the model sees the reason as a tool error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from langchain_core.tools import ToolException

from wigtn_agents.sap_sandbox.client import SapError
from wigtn_agents.sap_sandbox.fi import FiError, SapFi, shared_fi

Handler = Callable[[Optional[str], dict[str, Any]], dict[str, Any]]


def _guard(fi: Callable[[], SapFi], fn: Callable[[SapFi, dict[str, Any]], dict[str, Any]]) -> Handler:
    def handler(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return fn(fi(), payload)
        except (FiError, SapError) as exc:
            raise ToolException(str(exc)) from exc

    return handler


def _list(value: Any) -> Optional[list[str]]:
    if value in (None, "", []):
        return None
    return [str(v) for v in value] if isinstance(value, list) else [s.strip() for s in str(value).split(",")]


def sap_fi_routes(fi: Optional[SapFi] = None) -> dict[str, Handler]:
    """Route name → handler (the process-wide FI by default, built on first call)."""
    get = (lambda: fi) if fi is not None else shared_fi

    def co(p: dict[str, Any]) -> str:
        return str(p.get("company") or "")

    return {
        "sap-fi-company": _guard(get, lambda f, p: f.company(co(p))),
        "sap-fi-gl-accounts": _guard(
            get, lambda f, p: f.gl_accounts(co(p), accounts=_list(p.get("accounts")), search=p.get("search"))
        ),
        "sap-fi-journal-items": _guard(
            get,
            lambda f, p: f.journal_items(
                co(p),
                fiscal_year=p.get("fiscal_year"),
                fiscal_period=p.get("fiscal_period"),
                account=p.get("account"),
                document=p.get("document"),
                top=int(p.get("top") or 50),
            ),
        ),
        "sap-fi-account-balances": _guard(
            get,
            lambda f, p: f.account_balances(
                co(p),
                fiscal_year=str(p.get("fiscal_year") or ""),
                fiscal_period=p.get("fiscal_period"),
                accounts=_list(p.get("accounts")),
            ),
        ),
        "sap-fi-journal-post": _guard(
            get,
            lambda f, p: f.post_journal(
                co(p),
                text=str(p.get("text") or ""),
                lines=list(p.get("lines") or []),
                posting_date=p.get("posting_date"),
            ),
        ),
    }
