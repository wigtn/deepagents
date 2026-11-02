"""SAP FI on the read-only sandbox, with postings reflected in a local shadow ledger.

Reads (company codes, G/L accounts, journal entry items, account balances) go to the sandbox.
A posting cannot: the sandbox has no write access. ``post_journal`` therefore does what a real
posting would check first, against live master data (the company code exists, every account is
in its chart of accounts, debits equal credits), then records the entry in the shadow ledger
together with the request a real system would receive. Later reads add shadow lines, marked
``source: shadow``, so an agent sees its own posting the way it would on a writable system.

The shadow ledger is a JSONL file (``WIGTN_SAP_SHADOW_LOG``, default ``var/sap-shadow.jsonl``):
it survives a server restart and is the record of what an agent would have posted.

OData field names live in ``FIELDS``; they follow the S/4HANA Cloud APIs
``API_COMPANYCODE_SRV``, ``API_GLACCOUNTINCHARTOFACCOUNTS_SRV`` and ``API_JOURNALENTRYITEMBASIC_SRV``.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from wigtn_agents.sap_sandbox.client import SapError, SapSandboxClient, quote

SHADOW_LOG_ENV = "WIGTN_SAP_SHADOW_LOG"
MAX_ITEMS = 1000  # journal items read per balance query; a larger result is reported as truncated

COMPANY = ("API_COMPANYCODE_SRV", "A_CompanyCode")
GL_ACCOUNT = ("API_GLACCOUNTINCHARTOFACCOUNTS_SRV", "A_GLAccountInChartOfAccounts")
GL_TEXT = ("API_GLACCOUNTINCHARTOFACCOUNTS_SRV", "A_GLAccountText")
JOURNAL_ITEM = ("API_JOURNALENTRYITEMBASIC_SRV", "A_JournalEntryItemBasic")

FIELDS: dict[str, list[str]] = {
    "company": ["CompanyCode", "CompanyCodeName", "Country", "Currency", "ChartOfAccounts", "FiscalYearVariant"],
    "gl_account": ["ChartOfAccounts", "GLAccount", "GLAccountType", "IsBalanceSheetAccount"],
    "gl_text": ["ChartOfAccounts", "GLAccount", "Language", "GLAccountName", "GLAccountLongName"],
    "journal_item": [
        "CompanyCode",
        "FiscalYear",
        "FiscalPeriod",
        "AccountingDocument",
        "LedgerGLLineItem",
        "AccountingDocumentType",
        "PostingDate",
        "GLAccount",
        "DebitCreditCode",
        "AmountInCompanyCodeCurrency",
        "CompanyCodeCurrency",
        "DocumentItemText",
    ],
}


class FiError(ValueError):
    """The request is refused the way FI would refuse it (unknown account, unbalanced entry, ...)."""


def _amount(value: Any) -> Decimal:
    return Decimal(str(value or "0"))


def _num(value: Decimal) -> float | int:
    return int(value) if value == value.to_integral_value() else float(value)


class ShadowLedger:
    """Postings an agent made against the read-only sandbox, one JSON line each."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path(os.environ.get(SHADOW_LOG_ENV, "var/sap-shadow.jsonl"))
        self._lock = threading.Lock()

    def entries(self) -> list[dict[str, Any]]:
        """Every shadow posting, oldest first."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, entry: dict[str, Any]) -> None:
        """Record one posting."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        """Forget every shadow posting (tests, a fresh experiment)."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()


class SapFi:
    """FI reads on the sandbox and shadow postings."""

    def __init__(self, client: Optional[SapSandboxClient] = None, shadow: Optional[ShadowLedger] = None) -> None:
        self.client = client or SapSandboxClient()
        self.shadow = shadow or ShadowLedger()

    # -- master data -----------------------------------------------------------------------------

    def company(self, company: str) -> dict[str, Any]:
        """One company code: name, country, currency, chart of accounts."""
        rows = self.client.get(*COMPANY, filters=[f"CompanyCode eq {quote(company)}"], select=FIELDS["company"], top=1)
        if not rows:
            raise FiError(f"SAP에 없는 회사 코드입니다: {company}")
        return rows[0]

    def gl_accounts(
        self, company: str, *, accounts: Optional[list[str]] = None, search: Optional[str] = None, top: int = 50
    ) -> dict[str, Any]:
        """G/L accounts of the company's chart of accounts, with their English names."""
        chart = self.company(company)["ChartOfAccounts"]
        filters = [f"ChartOfAccounts eq {quote(chart)}", "Language eq 'EN'"]
        if accounts:
            filters.append("(" + " or ".join(f"GLAccount eq {quote(a)}" for a in accounts) + ")")
        if search:
            filters.append(f"substringof({quote(search)}, GLAccountName)")
        rows = self.client.get(*GL_TEXT, filters=filters, select=FIELDS["gl_text"], top=top)
        return {
            "company": company,
            "chart_of_accounts": chart,
            "accounts": [{"account": r["GLAccount"], "name": r.get("GLAccountName")} for r in rows],
        }

    # -- journal entries -------------------------------------------------------------------------

    def _shadow_items(self, company: str, year: Optional[str], period: Optional[str]) -> list[dict[str, Any]]:
        items = []
        for entry in self.shadow.entries():
            if entry["company"] != company or (year and entry["fiscal_year"] != year):
                continue
            if period and entry["fiscal_period"] != period:
                continue
            for n, line in enumerate(entry["lines"], start=1):
                items.append(
                    {
                        "document": entry["document"],
                        "item": n,
                        "posting_date": entry["posting_date"],
                        "fiscal_year": entry["fiscal_year"],
                        "fiscal_period": entry["fiscal_period"],
                        "account": line["account"],
                        "debit": line.get("debit", 0),
                        "credit": line.get("credit", 0),
                        "currency": entry["currency"],
                        "text": entry["text"],
                        "source": "shadow",
                    }
                )
        return items

    def journal_items(
        self,
        company: str,
        *,
        fiscal_year: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        account: Optional[str] = None,
        document: Optional[str] = None,
        top: int = 50,
    ) -> dict[str, Any]:
        """Journal entry line items, newest first, then this run's shadow lines."""
        filters = [f"CompanyCode eq {quote(company)}", "Ledger eq '0L'"]
        if fiscal_year:
            filters.append(f"FiscalYear eq {quote(fiscal_year)}")
        if fiscal_period:
            filters.append(f"FiscalPeriod eq {quote(fiscal_period.zfill(3))}")
        if account:
            filters.append(f"GLAccount eq {quote(account)}")
        if document:
            filters.append(f"AccountingDocument eq {quote(document)}")
        rows = self.client.get(
            *JOURNAL_ITEM, filters=filters, select=FIELDS["journal_item"], orderby="PostingDate desc", top=top
        )
        items = []
        for r in rows:
            amount = abs(_amount(r.get("AmountInCompanyCodeCurrency")))
            debit = r.get("DebitCreditCode") == "S"
            items.append(
                {
                    "document": r.get("AccountingDocument"),
                    "item": r.get("LedgerGLLineItem"),
                    "posting_date": r.get("PostingDate"),
                    "fiscal_year": r.get("FiscalYear"),
                    "fiscal_period": r.get("FiscalPeriod"),
                    "document_type": r.get("AccountingDocumentType"),
                    "account": r.get("GLAccount"),
                    "debit": _num(amount) if debit else 0,
                    "credit": 0 if debit else _num(amount),
                    "currency": r.get("CompanyCodeCurrency"),
                    "text": r.get("DocumentItemText"),
                    "source": "sap",
                }
            )
        shadow = [
            i
            for i in self._shadow_items(company, fiscal_year, fiscal_period)
            if (not account or i["account"] == account) and (not document or i["document"] == document)
        ]
        return {"company": company, "count": len(items) + len(shadow), "items": items + shadow}

    def account_balances(
        self,
        company: str,
        *,
        fiscal_year: str,
        fiscal_period: Optional[str] = None,
        accounts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Debit, credit and net per account for a fiscal year (or one period), SAP plus shadow."""
        filters = [f"CompanyCode eq {quote(company)}", "Ledger eq '0L'", f"FiscalYear eq {quote(fiscal_year)}"]
        if fiscal_period:
            filters.append(f"FiscalPeriod eq {quote(fiscal_period.zfill(3))}")
        if accounts:
            filters.append("(" + " or ".join(f"GLAccount eq {quote(a)}" for a in accounts) + ")")
        rows = self.client.get(
            *JOURNAL_ITEM,
            filters=filters,
            select=["GLAccount", "DebitCreditCode", "AmountInCompanyCodeCurrency", "CompanyCodeCurrency"],
            top=MAX_ITEMS,
        )
        debit: dict[str, Decimal] = defaultdict(Decimal)
        credit: dict[str, Decimal] = defaultdict(Decimal)
        for r in rows:
            target = debit if r.get("DebitCreditCode") == "S" else credit
            target[r["GLAccount"]] += abs(_amount(r.get("AmountInCompanyCodeCurrency")))
        shadow = [
            i
            for i in self._shadow_items(company, fiscal_year, fiscal_period)
            if not accounts or i["account"] in accounts
        ]
        for i in shadow:
            debit[i["account"]] += _amount(i["debit"])
            credit[i["account"]] += _amount(i["credit"])
        names = sorted(set(debit) | set(credit))
        return {
            "company": company,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "currency": rows[0].get("CompanyCodeCurrency") if rows else None,
            "accounts": [
                {
                    "account": a,
                    "debit": _num(debit[a]),
                    "credit": _num(credit[a]),
                    "net": _num(debit[a] - credit[a]),
                }
                for a in names
            ],
            "sap_items_read": len(rows),
            "shadow_items": len(shadow),
            "truncated": len(rows) >= MAX_ITEMS,
        }

    # -- shadow posting --------------------------------------------------------------------------

    def post_journal(
        self,
        company: str,
        *,
        text: str,
        lines: list[dict[str, Any]],
        posting_date: Optional[str] = None,
        document_type: str = "SA",
    ) -> dict[str, Any]:
        """Check a journal entry against live master data and record it in the shadow ledger."""
        if len(lines) < 2:
            raise FiError("전표에는 차변과 대변 줄이 모두 있어야 합니다")
        day = posting_date or date.today().isoformat()
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise FiError(f"전기일은 YYYY-MM-DD 형식이어야 합니다: {day}") from exc
        header = self.company(company)
        clean = []
        for line in lines:
            account = str(line.get("account", "")).strip()
            d, c = _amount(line.get("debit")), _amount(line.get("credit"))
            if not account or (d > 0) == (c > 0) or d < 0 or c < 0:
                raise FiError(f"한 줄에는 계정과, 차변이나 대변 중 하나만 양수로 적습니다: {line}")
            clean.append({"account": account, **({"debit": _num(d)} if d else {"credit": _num(c)})})
        total_d = sum((_amount(x.get("debit")) for x in clean), Decimal(0))
        total_c = sum((_amount(x.get("credit")) for x in clean), Decimal(0))
        if total_d != total_c:
            raise FiError(f"차변 합계 {_num(total_d)}와 대변 합계 {_num(total_c)}가 다릅니다")
        wanted = sorted({str(x["account"]) for x in clean})
        known = {a["account"] for a in self.gl_accounts(company, accounts=wanted, top=len(wanted))["accounts"]}
        missing = [a for a in wanted if a not in known]
        if missing:
            raise FiError(f"계정과목표 {header['ChartOfAccounts']}에 없는 계정입니다: {missing}")
        year, period = day[:4], day[5:7].zfill(3)  # calendar fiscal year (variant K4)
        number = len([e for e in self.shadow.entries() if e["company"] == company and e["fiscal_year"] == year]) + 1
        entry = {
            "document": f"SHADOW-{company}-{year}-{number:04d}",
            "company": company,
            "fiscal_year": year,
            "fiscal_period": period,
            "posting_date": day,
            "document_type": document_type,
            "currency": header.get("Currency"),
            "text": text,
            "lines": clean,
            # What a writable S/4HANA system would receive (JournalEntryBulkCreationRequest, one entry).
            "sap_request": {
                "CompanyCode": company,
                "AccountingDocumentType": document_type,
                "DocumentDate": day,
                "PostingDate": day,
                "DocumentHeaderText": text[:25],
                "Item": [
                    {
                        "GLAccount": x["account"],
                        "AmountInTransactionCurrency": {
                            "currencyCode": header.get("Currency"),
                            "value": str(x.get("debit") or -Decimal(str(x["credit"]))),
                        },
                        "DebitCreditCode": "S" if "debit" in x else "H",
                    }
                    for x in clean
                ],
            },
        }
        self.shadow.append(entry)
        return {
            "company": company,
            "document": entry["document"],
            "reflected_in": "shadow ledger",
            "note": "SAP 샌드박스는 읽기 전용이라 SAP에는 바뀐 것이 없습니다. 이후 조회에는 source=shadow로 나타납니다",
            "posting_date": day,
            "amount": _num(total_d),
            "currency": header.get("Currency"),
        }


_shared: dict[str, Optional[SapFi]] = {"fi": None}


def shared_fi() -> SapFi:
    """The process-wide FI (one sandbox client cache, one shadow ledger)."""
    fi = _shared["fi"]
    if fi is None:
        fi = _shared["fi"] = SapFi()
    return fi


__all__ = ["FiError", "SapError", "SapFi", "ShadowLedger", "shared_fi"]
