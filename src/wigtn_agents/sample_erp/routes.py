"""Gateway routes of the sample ERP: one route per tool asset in ``catalog/assets/tool.erp-*.yaml``.

Every route takes ``company`` (the scope guard writes it from the run context; the model's value
never reaches here) and answers a JSON object. A refused request raises ``ToolException`` so
the model sees the ERP's reason as a tool error and the run record marks the call failed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from langchain_core.tools import ToolException

from wigtn_agents.sample_erp.system import ErpError, SampleErp, shared_erp

Handler = Callable[[Optional[str], dict[str, Any]], dict[str, Any]]


def _guard(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Handler:
    def handler(operation: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return fn(payload)
        except ErpError as exc:
            raise ToolException(str(exc)) from exc

    return handler


def erp_routes(erp: Optional[SampleErp] = None) -> dict[str, Handler]:
    """Route name → handler over ``erp`` (the process-wide ERP by default)."""
    system = erp or shared_erp()

    def company(p: dict[str, Any]) -> str:
        return str(p.get("company") or "")

    return {
        "erp-period-status": _guard(lambda p: system.period_status(company(p))),
        "erp-po-read": _guard(
            lambda p: system.purchase_orders_report(company(p), po=p.get("po"), vendor=p.get("vendor"))
        ),
        "erp-invoice-read": _guard(
            lambda p: system.vendor_invoices_report(company(p), invoice=p.get("invoice"), vendor=p.get("vendor"))
        ),
        "erp-three-way-match": _guard(lambda p: system.three_way_match(company(p), invoice=p.get("invoice"))),
        "erp-stock-read": _guard(lambda p: system.stock_report(company(p), material=p.get("material"))),
        "erp-ar-aging": _guard(lambda p: system.ar_aging(company(p), customer=p.get("customer"))),
        "erp-trial-balance": _guard(lambda p: system.trial_balance(company(p), period=p.get("period"))),
        "erp-close-checklist": _guard(lambda p: system.close_checklist(company(p), period=p.get("period"))),
        "erp-journal-post": _guard(
            lambda p: system.post_journal(
                company(p), period=p.get("period"), text=str(p.get("text") or ""), lines=list(p.get("lines") or [])
            )
        ),
        "erp-payment-block": _guard(
            lambda p: system.set_payment_block(
                company(p),
                invoice=str(p.get("invoice") or ""),
                block=p.get("block") is not False,
                reason=str(p.get("reason") or ""),
            )
        ),
        "erp-requisition-create": _guard(
            lambda p: system.create_requisition(
                company(p), lines=list(p.get("lines") or []), note=str(p.get("note") or "")
            )
        ),
    }
