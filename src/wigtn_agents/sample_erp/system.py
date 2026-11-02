"""The sample ERP: state per company code, read reports and three write operations.

Reads compute their answers from the state (three-way match, reorder need, aging, trial
balance, close checklist), so an agent's answer can be checked against numbers. Writes change
the state and are validated the way an ERP would refuse them: a journal entry must balance, use
known accounts and land in an open period; a payment block names an existing invoice; a
requisition names known materials. A refused write raises ``ErpError``; the tool turns it into a
tool error the model sees.

Purchasing reaches the ledger the usual way: a goods receipt debits inventory and credits GR/IR
at the PO price, a vendor invoice clears GR/IR at the PO price, books the price difference to
inventory and credits accounts payable at the invoiced price.
"""

from __future__ import annotations

import copy
import threading
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Optional

from wigtn_agents.sample_erp import data

AGING_BUCKETS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("current", -10_000, 0),
    ("1-30", 1, 30),
    ("31-60", 31, 60),
    ("61-90", 61, 90),
    ("90+", 91, None),
)


class ErpError(ValueError):
    """The ERP refuses the request (unknown key, closed period, unbalanced entry, ...)."""


def _days(later: str, earlier: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _period(day: str) -> str:
    return day[:7]


class SampleErp:
    """In-memory ERP. One instance holds every company code; methods take the company code first."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Back to the seed data."""
        with self._lock:
            self.business_date = data.BUSINESS_DATE
            self.periods = copy.deepcopy(data.PERIODS)
            self.vendors = copy.deepcopy(data.VENDORS)
            self.customers = copy.deepcopy(data.CUSTOMERS)
            self.materials = copy.deepcopy(data.MATERIALS)
            self.purchase_orders = copy.deepcopy(data.PURCHASE_ORDERS)
            self.goods_receipts = copy.deepcopy(data.GOODS_RECEIPTS)
            self.vendor_invoices = copy.deepcopy(data.VENDOR_INVOICES)
            self.customer_invoices = copy.deepcopy(data.CUSTOMER_INVOICES)
            self.opening = copy.deepcopy(data.OPENING_BALANCES)
            self.journal = copy.deepcopy(data.JOURNAL_ENTRIES)
            self.close_items = copy.deepcopy(data.RECURRING_CLOSE_ITEMS)
            self.requisitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
            self.blocks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    # -- helpers ---------------------------------------------------------------------------------

    def _company(self, company: Optional[str]) -> str:
        if company not in data.COMPANIES:
            raise ErpError(f"없는 회사 코드입니다: {company}")
        return company

    def _received(self, company: str, po: str) -> int:
        return sum(g["qty"] for g in self.goods_receipts[company] if g["po"] == po)

    def _invoices_for(self, company: str, po: str) -> list[tuple[str, dict[str, Any]]]:
        rows = [(no, inv) for no, inv in self.vendor_invoices[company].items() if inv["po"] == po]
        return sorted(rows, key=lambda r: (r[1]["date"], r[0]))

    def _open_po_qty(self, company: str, material: str) -> int:
        return sum(
            max(po["qty"] - self._received(company, no), 0)
            for no, po in self.purchase_orders[company].items()
            if po["material"] == material
        )

    # -- reads -----------------------------------------------------------------------------------

    def period_status(self, company: str) -> dict[str, Any]:
        """Business date, the open period and every period's status."""
        company = self._company(company)
        periods = self.periods[company]
        return {
            "company": company,
            "company_name": data.COMPANIES[company]["name"],
            "currency": data.COMPANIES[company]["currency"],
            "business_date": self.business_date,
            "open_period": next((p for p, s in periods.items() if s == "open"), None),
            "periods": periods,
        }

    def purchase_orders_report(
        self, company: str, *, po: Optional[str] = None, vendor: Optional[str] = None
    ) -> dict[str, Any]:
        """Purchase orders with ordered, received and invoiced quantities."""
        company = self._company(company)
        rows: list[dict[str, Any]] = []
        for no, order in self.purchase_orders[company].items():
            if (po and no != po) or (vendor and order["vendor"] != vendor):
                continue
            invoiced = sum(inv["qty"] for _, inv in self._invoices_for(company, no))
            received = self._received(company, no)
            rows.append(
                {
                    "po": no,
                    **order,
                    "vendor_name": self.vendors[company][order["vendor"]]["name"],
                    "received_qty": received,
                    "invoiced_qty": invoiced,
                    "open_qty": max(order["qty"] - received, 0),
                    "value": order["qty"] * order["price"],
                }
            )
        if po and not rows:
            raise ErpError(f"없는 구매오더입니다: {po}")
        return {"company": company, "count": len(rows), "purchase_orders": rows}

    def vendor_invoices_report(
        self, company: str, *, invoice: Optional[str] = None, vendor: Optional[str] = None
    ) -> dict[str, Any]:
        """Vendor invoices with amount, due date and payment block."""
        company = self._company(company)
        rows: list[dict[str, Any]] = []
        for no, inv in sorted(self.vendor_invoices[company].items(), key=lambda r: (r[1]["date"], r[0])):
            if (invoice and no != invoice) or (vendor and inv["vendor"] != vendor):
                continue
            terms = self.vendors[company][inv["vendor"]]["terms_days"]
            rows.append(
                {
                    "invoice": no,
                    **inv,
                    "amount": inv["qty"] * inv["price"],
                    "due": (date.fromisoformat(inv["date"]) + timedelta(days=terms)).isoformat(),
                    "payment_block": self.blocks[company].get(no),
                }
            )
        if invoice and not rows:
            raise ErpError(f"없는 매입송장입니다: {invoice}")
        return {"company": company, "count": len(rows), "invoices": rows}

    def three_way_match(self, company: str, *, invoice: Optional[str] = None) -> dict[str, Any]:
        """Match every vendor invoice (or one) against its PO and goods receipts.

        Checks, per invoice: price against the PO price (tolerance ``PRICE_TOLERANCE_PCT``),
        quantity invoiced so far against quantity received, and an earlier invoice from the same
        vendor for the same PO, quantity and price (possible duplicate).
        """
        company = self._company(company)
        if invoice and invoice not in self.vendor_invoices[company]:
            raise ErpError(f"없는 매입송장입니다: {invoice}")
        results: list[dict[str, Any]] = []
        for po_no in self.purchase_orders[company]:
            order = self.purchase_orders[company][po_no]
            received = self._received(company, po_no)
            invoiced_so_far = 0
            seen: list[tuple[str, dict[str, Any]]] = []
            for no, inv in self._invoices_for(company, po_no):
                invoiced_so_far += inv["qty"]
                issues = []
                variance_pct = round((inv["price"] - order["price"]) / order["price"] * 100, 2)
                if variance_pct > data.PRICE_TOLERANCE_PCT:
                    issues.append(
                        {
                            "type": "price_variance",
                            "po_price": order["price"],
                            "invoice_price": inv["price"],
                            "variance_pct": variance_pct,
                            "amount": (inv["price"] - order["price"]) * inv["qty"],
                        }
                    )
                if invoiced_so_far > received:
                    over = min(invoiced_so_far - received, inv["qty"])
                    issues.append(
                        {
                            "type": "quantity_variance",
                            "received_qty": received,
                            "invoiced_qty_to_date": invoiced_so_far,
                            "over_qty": over,
                            "amount": over * inv["price"],
                        }
                    )
                twin = next(
                    (
                        prev
                        for prev, p in seen
                        if p["vendor"] == inv["vendor"] and p["qty"] == inv["qty"] and p["price"] == inv["price"]
                    ),
                    None,
                )
                if twin:
                    issues.append({"type": "possible_duplicate", "of_invoice": twin})
                seen.append((no, inv))
                if invoice and no != invoice:
                    continue
                results.append(
                    {
                        "invoice": no,
                        "po": po_no,
                        "vendor": inv["vendor"],
                        "amount": inv["qty"] * inv["price"],
                        "status": "exception" if issues else "matched",
                        "issues": issues,
                        "payment_block": self.blocks[company].get(no),
                    }
                )
        exceptions = [r for r in results if r["status"] == "exception"]
        return {
            "company": company,
            "price_tolerance_pct": data.PRICE_TOLERANCE_PCT,
            "checked": len(results),
            "exceptions": len(exceptions),
            "results": sorted(results, key=lambda r: r["invoice"]),
        }

    def stock_report(self, company: str, *, material: Optional[str] = None) -> dict[str, Any]:
        """Stock per material and the reorder need.

        available = on hand - reserved + open PO quantity. Below the reorder point, the
        suggested quantity brings available up to reorder point + safety stock, rounded up to
        the minimum order quantity.
        """
        company = self._company(company)
        rows: list[dict[str, Any]] = []
        for no, mat in self.materials[company].items():
            if material and no != material:
                continue
            open_po = self._open_po_qty(company, no)
            open_pr = sum(
                line["qty"] for pr in self.requisitions[company] for line in pr["lines"] if line["material"] == no
            )
            available = mat["on_hand"] - mat["reserved"] + open_po + open_pr
            below = available < mat["reorder_point"]
            need = max(mat["reorder_point"] + mat["safety_stock"] - available, 0) if below else 0
            moq = mat["min_order_qty"]
            suggested = -(-need // moq) * moq if need else 0
            rows.append(
                {
                    "material": no,
                    "description": mat["description"],
                    "uom": mat["uom"],
                    "on_hand": mat["on_hand"],
                    "reserved": mat["reserved"],
                    "open_po_qty": open_po,
                    "open_requisition_qty": open_pr,
                    "available": available,
                    "reorder_point": mat["reorder_point"],
                    "safety_stock": mat["safety_stock"],
                    "below_reorder_point": below,
                    "suggested_qty": suggested,
                    "suggested_value": suggested * mat["unit_cost"],
                    "vendor": mat["vendor"],
                    "lead_time_days": mat["lead_time_days"],
                }
            )
        if material and not rows:
            raise ErpError(f"없는 자재입니다: {material}")
        return {"company": company, "as_of": self.business_date, "materials": rows}

    def ar_aging(self, company: str, *, customer: Optional[str] = None) -> dict[str, Any]:
        """Open receivables by customer and days past due, with credit limit use."""
        company = self._company(company)
        per_customer: dict[str, dict[str, Any]] = {}
        for no, inv in self.customer_invoices[company].items():
            if customer and inv["customer"] != customer:
                continue
            open_amount = inv["amount"] - inv["paid"]
            if open_amount <= 0:
                continue
            overdue = _days(self.business_date, inv["due"])
            bucket = next(name for name, lo, hi in AGING_BUCKETS if overdue >= lo and (hi is None or overdue <= hi))
            info = self.customers[company][inv["customer"]]
            row = per_customer.setdefault(
                inv["customer"],
                {
                    "customer": inv["customer"],
                    "name": info["name"],
                    "credit_limit": info["credit_limit"],
                    "buckets": {name: 0 for name, _, _ in AGING_BUCKETS},
                    "items": [],
                },
            )
            row["buckets"][bucket] += open_amount
            row["items"].append(
                {"invoice": no, "open": open_amount, "due": inv["due"], "days_overdue": max(overdue, 0)}
            )
        rows: list[dict[str, Any]] = []
        for row in per_customer.values():
            total = sum(row["buckets"].values())
            rows.append(
                {
                    **row,
                    "total_open": total,
                    "overdue": total - row["buckets"]["current"],
                    "over_credit_limit": max(total - row["credit_limit"], 0),
                }
            )
        totals = {name: sum(r["buckets"][name] for r in rows) for name, _, _ in AGING_BUCKETS}
        return {"company": company, "as_of": self.business_date, "customers": rows, "totals": totals}

    def _postings(self, company: str, period: str) -> list[dict[str, Any]]:
        """Every ledger line of the period: journal entries plus the purchasing flow."""
        lines: list[dict[str, Any]] = []
        for entry in self.journal[company]:
            if _period(entry["date"]) == period:
                lines.extend({**line, "doc": entry["doc"]} for line in entry["lines"])
        for gr in self.goods_receipts[company]:
            if _period(gr["date"]) != period:
                continue
            value = gr["qty"] * self.purchase_orders[company][gr["po"]]["price"]
            lines += [
                {"account": "1300", "debit": value, "doc": gr["gr"]},
                {"account": "2150", "credit": value, "doc": gr["gr"]},
            ]
        for no, inv in self.vendor_invoices[company].items():
            if _period(inv["date"]) != period:
                continue
            at_po = inv["qty"] * self.purchase_orders[company][inv["po"]]["price"]
            billed = inv["qty"] * inv["price"]
            lines += [
                {"account": "2150", "debit": at_po, "doc": no},
                {"account": "2100", "credit": billed, "doc": no},
            ]
            if billed != at_po:
                lines.append({"account": "1300", "debit": billed - at_po, "doc": no})
        return lines

    def trial_balance(self, company: str, *, period: Optional[str] = None) -> dict[str, Any]:
        """Opening balance, period debits and credits and closing balance per account."""
        company = self._company(company)
        period = period or _period(self.business_date)
        if period not in self.periods[company]:
            raise ErpError(f"없는 회계기간입니다: {period}")
        debit: dict[str, int] = defaultdict(int)
        credit: dict[str, int] = defaultdict(int)
        for line in self._postings(company, period):
            debit[line["account"]] += line.get("debit", 0)
            credit[line["account"]] += line.get("credit", 0)
        opening = self.opening[company] if period == _period(self.business_date) else {}
        accounts = sorted(set(opening) | set(debit) | set(credit))
        rows: list[dict[str, Any]] = [
            {
                "account": acc,
                "name": data.ACCOUNTS[acc]["name"],
                "opening": opening.get(acc, 0),
                "debit": debit[acc],
                "credit": credit[acc],
                "closing": opening.get(acc, 0) + debit[acc] - credit[acc],
            }
            for acc in accounts
        ]
        total_debit, total_credit = sum(debit.values()), sum(credit.values())
        return {
            "company": company,
            "period": period,
            "accounts": rows,
            "total_debit": total_debit,
            "total_credit": total_credit,
            "balanced": total_debit == total_credit and sum(r["closing"] for r in rows) == 0,
        }

    def gr_ir_open_items(self, company: str) -> list[dict[str, Any]]:
        """Per PO: received value minus invoiced value at the PO price. Positive = received, not invoiced."""
        rows: list[dict[str, Any]] = []
        for no, order in self.purchase_orders[company].items():
            received = self._received(company, no)
            invoiced = sum(inv["qty"] for _, inv in self._invoices_for(company, no))
            balance = (received - invoiced) * order["price"]
            if balance:
                rows.append(
                    {
                        "po": no,
                        "vendor": order["vendor"],
                        "received_qty": received,
                        "invoiced_qty": invoiced,
                        "balance": balance,
                        "kind": "received_not_invoiced" if balance > 0 else "invoiced_not_received",
                    }
                )
        return rows

    def close_checklist(self, company: str, *, period: Optional[str] = None) -> dict[str, Any]:
        """Month-end checks for a period: status, recurring postings, GR/IR, AP exceptions, AR vs ledger, balance."""
        company = self._company(company)
        period = period or _period(self.business_date)
        status = self.periods[company].get(period)
        if status is None:
            raise ErpError(f"없는 회계기간입니다: {period}")
        checks: list[dict[str, Any]] = [{"id": "period-open", "ok": status == "open", "detail": f"{period}: {status}"}]
        for item in self.close_items[company]:
            posted = next(
                (
                    entry["doc"]
                    for entry in self.journal[company]
                    if _period(entry["date"]) == period and self._settles(entry, item)
                ),
                None,
            )
            check: dict[str, Any] = {"id": item["id"], "ok": posted is not None, "detail": item["text"]}
            if posted:
                check["posted_by"] = posted
            elif item["proposal"]:
                check["proposal"] = {
                    "lines": [
                        {"account": item["proposal"]["debit"], "debit": item["amount"]},
                        {"account": item["proposal"]["credit"], "credit": item["amount"]},
                    ],
                    "amount": item["amount"],
                }
            checks.append(check)
        gr_ir = self.gr_ir_open_items(company)
        checks.append(
            {
                "id": "gr-ir-review",
                "ok": all(r["kind"] == "received_not_invoiced" for r in gr_ir),
                "detail": "GR/IR items invoiced but not received need follow-up before close",
                "items": gr_ir,
            }
        )
        match = self.three_way_match(company)
        unblocked = [r["invoice"] for r in match["results"] if r["status"] == "exception" and not r["payment_block"]]
        checks.append(
            {
                "id": "ap-exceptions-blocked",
                "ok": not unblocked,
                "detail": "Invoices with match exceptions should carry a payment block",
                "unblocked": unblocked,
            }
        )
        tb = self.trial_balance(company, period=period)
        ar_ledger = next((r["closing"] for r in tb["accounts"] if r["account"] == "1200"), 0)
        ar_sub = sum(c["total_open"] for c in self.ar_aging(company)["customers"])
        checks.append(
            {"id": "ar-reconciled", "ok": ar_ledger == ar_sub, "detail": f"ledger {ar_ledger:,} / subledger {ar_sub:,}"}
        )
        checks.append({"id": "trial-balance", "ok": tb["balanced"], "detail": f"debit {tb['total_debit']:,}"})
        return {
            "company": company,
            "period": period,
            "ready_to_close": all(c["ok"] for c in checks),
            "open_checks": [c["id"] for c in checks if not c["ok"]],
            "checks": checks,
        }

    @staticmethod
    def _settles(entry: dict[str, Any], item: dict[str, Any]) -> bool:
        debit = sum(line.get("debit", 0) for line in entry["lines"] if line["account"] == item["match"]["debit"])
        credit = sum(line.get("credit", 0) for line in entry["lines"] if line["account"] == item["match"]["credit"])
        return bool(debit == item["amount"] and credit == item["amount"])

    # -- writes ----------------------------------------------------------------------------------

    def post_journal(
        self, company: str, *, period: Optional[str], text: str, lines: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Post a journal entry on the last day of an open period."""
        company = self._company(company)
        period = period or _period(self.business_date)
        if self.periods[company].get(period) != "open":
            raise ErpError(f"{period}는 열린 회계기간이 아닙니다 ({self.periods[company].get(period)})")
        if len(lines) < 2:
            raise ErpError("전표에는 차변과 대변 줄이 모두 있어야 합니다")
        clean: list[dict[str, Any]] = []
        for line in lines:
            account = str(line.get("account", ""))
            if account not in data.ACCOUNTS:
                raise ErpError(f"없는 계정입니다: {account}")
            debit, credit = int(line.get("debit") or 0), int(line.get("credit") or 0)
            if (debit > 0) == (credit > 0) or debit < 0 or credit < 0:
                raise ErpError(f"한 줄에는 차변이나 대변 중 하나만 양수로 적습니다: {line}")
            clean.append({"account": account, **({"debit": debit} if debit else {"credit": credit})})
        total_debit = sum(line.get("debit", 0) for line in clean)
        total_credit = sum(line.get("credit", 0) for line in clean)
        if total_debit != total_credit:
            raise ErpError(f"차변 합계 {total_debit:,}와 대변 합계 {total_credit:,}가 다릅니다")
        with self._lock:
            count = len([e for e in self.journal[company] if _period(e["date"]) == period])
            doc = f"JE-{company}-{period}-{count + 1:03d}"
            posting_date = self.business_date if _period(self.business_date) == period else f"{period}-28"
            self.journal[company].append({"doc": doc, "date": posting_date, "text": text, "lines": clean})
        return {"company": company, "document": doc, "period": period, "date": posting_date, "amount": total_debit}

    def set_payment_block(self, company: str, *, invoice: str, block: bool, reason: str) -> dict[str, Any]:
        """Set or release the payment block of a vendor invoice."""
        company = self._company(company)
        if invoice not in self.vendor_invoices[company]:
            raise ErpError(f"없는 매입송장입니다: {invoice}")
        if block and not reason.strip():
            raise ErpError("지급 보류에는 사유가 필요합니다")
        with self._lock:
            if block:
                self.blocks[company][invoice] = {"reason": reason, "date": self.business_date}
            else:
                self.blocks[company].pop(invoice, None)
        return {"company": company, "invoice": invoice, "payment_block": self.blocks[company].get(invoice)}

    def create_requisition(self, company: str, *, lines: list[dict[str, Any]], note: str = "") -> dict[str, Any]:
        """Create a purchase requisition; delivery date = business date + the longest lead time."""
        company = self._company(company)
        if not lines:
            raise ErpError("구매요청에는 자재 줄이 하나 이상 있어야 합니다")
        clean: list[dict[str, Any]] = []
        for line in lines:
            material = str(line.get("material", ""))
            if material not in self.materials[company]:
                raise ErpError(f"없는 자재입니다: {material}")
            qty = int(line.get("qty") or 0)
            if qty <= 0:
                raise ErpError(f"수량은 양수여야 합니다: {line}")
            clean.append({"material": material, "qty": qty})
        lead = max(self.materials[company][line["material"]]["lead_time_days"] for line in clean)
        with self._lock:
            number = f"PR-{company}-{len(self.requisitions[company]) + 1:04d}"
            pr = {
                "requisition": number,
                "lines": clean,
                "note": note,
                "delivery_date": (date.fromisoformat(self.business_date) + timedelta(days=lead)).isoformat(),
                "value": sum(line["qty"] * self.materials[company][line["material"]]["unit_cost"] for line in clean),
            }
            self.requisitions[company].append(pr)
        return {"company": company, **pr}


_shared: dict[str, Optional[SampleErp]] = {"erp": None}


def shared_erp() -> SampleErp:
    """The process-wide ERP: a server's runs see each other's writes until the process restarts."""
    erp = _shared["erp"]
    if erp is None:
        erp = _shared["erp"] = SampleErp()
    return erp
