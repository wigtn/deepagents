"""SAP FI on the read-only sandbox: reads, the shadow posting, and the agent end to end (mock sandbox)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from langchain_core.messages import AIMessage

from tests.conftest import SHIPPED_CATALOG_DIR, scripted_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import default_context
from wigtn_agents.runner import run_local
from wigtn_agents.sap_sandbox import FiError, SapError, SapFi, SapSandboxClient, ShadowLedger, sap_fi_routes
from wigtn_agents.sap_sandbox import fi as fi_module

COMPANY = {
    "CompanyCode": "1010",
    "CompanyCodeName": "Company Code 1010",
    "Country": "DE",
    "Currency": "EUR",
    "ChartOfAccounts": "YCOA",
    "FiscalYearVariant": "K4",
}
GL_TEXT = [
    {"ChartOfAccounts": "YCOA", "GLAccount": "11001000", "Language": "EN", "GLAccountName": "Bank 1"},
    {"ChartOfAccounts": "YCOA", "GLAccount": "61002000", "Language": "EN", "GLAccountName": "Rent Expense"},
]
ITEMS = [
    {
        "CompanyCode": "1010",
        "FiscalYear": "2023",
        "FiscalPeriod": "007",
        "AccountingDocument": "100000123",
        "LedgerGLLineItem": "000001",
        "AccountingDocumentType": "SA",
        "PostingDate": "/Date(1689811200000)/",
        "GLAccount": "61002000",
        "DebitCreditCode": "S",
        "AmountInCompanyCodeCurrency": "500.00",
        "CompanyCodeCurrency": "EUR",
        "DocumentItemText": "July rent",
    },
    {
        "CompanyCode": "1010",
        "FiscalYear": "2023",
        "FiscalPeriod": "007",
        "AccountingDocument": "100000123",
        "LedgerGLLineItem": "000002",
        "AccountingDocumentType": "SA",
        "PostingDate": "/Date(1689811200000)/",
        "GLAccount": "11001000",
        "DebitCreditCode": "H",
        "AmountInCompanyCodeCurrency": "-500.00",
        "CompanyCodeCurrency": "EUR",
        "DocumentItemText": "July rent",
    },
]


def fake_sandbox(seen: list[dict[str, Any]]) -> httpx.MockTransport:
    """Answers the three services the way the sandbox does (OData v2 JSON, ``d.results``)."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers.get("APIKey") != "k-test":
            return httpx.Response(401, json={"fault": {"faultstring": "Failed to resolve API Key"}})
        url = urlparse(str(request.url))
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        seen.append({"path": url.path, **query})
        flt = query.get("$filter", "")
        if url.path.endswith("/A_CompanyCode"):
            rows = [COMPANY] if "'1010'" in flt else []
        elif url.path.endswith("/A_GLAccountText"):
            rows = [r for r in GL_TEXT if "GLAccount eq" not in flt or f"'{r['GLAccount']}'" in flt]
        elif url.path.endswith("/A_JournalEntryItemBasic"):
            rows = [r for r in ITEMS if f"FiscalYear eq '{r['FiscalYear']}'" in flt or "FiscalYear" not in flt]
        else:
            return httpx.Response(404, text="no such service")
        return httpx.Response(200, json={"d": {"results": [{"__metadata": {}, **r} for r in rows]}})

    return httpx.MockTransport(handle)


@pytest.fixture
def seen() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def fi(seen: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SapFi:
    client = SapSandboxClient(api_key="k-test", transport=fake_sandbox(seen))
    sap = SapFi(client=client, shadow=ShadowLedger(tmp_path / "shadow.jsonl"))
    monkeypatch.setitem(fi_module._shared, "fi", sap)
    return sap


def test_키가_없으면_호출_전에_이유를_알린다() -> None:
    with pytest.raises(SapError, match="SAP_SANDBOX_API_KEY"):
        SapSandboxClient(transport=fake_sandbox([])).get("API_COMPANYCODE_SRV", "A_CompanyCode")


def test_조회는_OData_필터를_쓰고_날짜와_차대를_풀어_준다(fi: SapFi, seen: list[dict[str, Any]]) -> None:
    items = fi.journal_items("1010", fiscal_year="2023", fiscal_period="7")["items"]
    assert (
        seen[-1]["$filter"]
        == "CompanyCode eq '1010' and Ledger eq '0L' and FiscalYear eq '2023' and FiscalPeriod eq '007'"
    )
    assert seen[-1]["$orderby"] == "PostingDate desc" and seen[-1]["$format"] == "json"
    assert items[0]["posting_date"] == "2023-07-20" and items[0]["debit"] == 500 and items[1]["credit"] == 500
    fi.journal_items("1010", fiscal_year="2023", fiscal_period="7")
    assert fi.client.requests == 1  # same URL: answered from the cache


def test_반영은_실제_마스터로_검증하고_그림자_원장에만_남는다(fi: SapFi) -> None:
    with pytest.raises(FiError, match="없는 계정"):
        fi.post_journal(
            "1010",
            text="x",
            posting_date="2023-07-31",
            lines=[{"account": "99999999", "debit": 1}, {"account": "11001000", "credit": 1}],
        )
    with pytest.raises(FiError, match="다릅니다"):
        fi.post_journal(
            "1010", text="x", lines=[{"account": "61002000", "debit": 2}, {"account": "11001000", "credit": 1}]
        )
    with pytest.raises(FiError, match="없는 회사 코드"):
        fi.post_journal(
            "9999", text="x", lines=[{"account": "61002000", "debit": 1}, {"account": "11001000", "credit": 1}]
        )

    done = fi.post_journal(
        "1010",
        text="August rent",
        posting_date="2023-07-31",
        lines=[{"account": "61002000", "debit": 1000}, {"account": "11001000", "credit": 1000}],
    )
    assert done["document"] == "SHADOW-1010-2023-0001" and done["reflected_in"] == "shadow ledger"
    entry = fi.shadow.entries()[0]
    assert entry["sap_request"]["Item"][1] == {
        "GLAccount": "11001000",
        "AmountInTransactionCurrency": {"currencyCode": "EUR", "value": "-1000"},
        "DebitCreditCode": "H",
    }
    balances = {a["account"]: a for a in fi.account_balances("1010", fiscal_year="2023")["accounts"]}
    assert balances["61002000"] == {"account": "61002000", "debit": 1500, "credit": 0, "net": 1500}
    shadow_lines = [i for i in fi.journal_items("1010", fiscal_year="2023")["items"] if i["source"] == "shadow"]
    assert [i["account"] for i in shadow_lines] == ["61002000", "11001000"]


def test_라우트는_FI_거부를_도구_오류로_낸다(fi: SapFi) -> None:
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException, match="없는 회사 코드"):
        sap_fi_routes(fi)["sap-fi-company"](None, {"company": "9999"})
    assert sap_fi_routes(fi)["sap-fi-gl-accounts"](None, {"company": "1010", "accounts": "61002000"})["accounts"] == [
        {"account": "61002000", "name": "Rent Expense"}
    ]


async def test_SAP_FI_에이전트는_승인_후에만_그림자_원장에_반영한다(fi: SapFi, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIGTN_AGENT_CATALOG_DIR", str(SHIPPED_CATALOG_DIR))
    catalog = Catalog.load(SHIPPED_CATALOG_DIR / "assets", SHIPPED_CATALOG_DIR / "manifests")
    post = {
        "text": "rent",
        "posting_date": "2023-07-31",
        "company": "2000",
        "lines": [{"account": "61002000", "debit": 1000}, {"account": "11001000", "credit": 1000}],
    }

    def replies() -> Any:
        return scripted_model(
            AIMessage(content="", tool_calls=[{"name": "sap-fi-gl-accounts", "args": {"search": "Rent"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "sap-fi-journal-post", "args": post, "id": "c2"}]),
            AIMessage(content="반영 요청을 처리했습니다"),
        )

    ctx = default_context("sap-fi-accountant", entity="1010", principal="u-1")
    stopped = await run_local("sap-fi-accountant", "임차료 전표", ctx, catalog=catalog, model=replies(), sink=[])
    assert stopped.status == "interrupted" and stopped.pending[0]["name"] == "sap-fi-journal-post"
    assert fi.shadow.entries() == []

    done = await run_local(
        "sap-fi-accountant", "임차료 전표", ctx, catalog=catalog, model=replies(), sink=[], approve=True
    )
    assert done.status == "success"
    entries = fi.shadow.entries()
    assert len(entries) == 1 and entries[0]["company"] == "1010"  # the model's "2000" never reached FI
    assert json.dumps(entries[0]["lines"]) == json.dumps(post["lines"])
