"""The sample ERP's answers are the ground truth the ERP catalog's eval cases check against."""

from __future__ import annotations

import pytest
from langchain_core.tools import ToolException

from wigtn_agents.sample_erp import ErpError, SampleErp, erp_routes


@pytest.fixture
def erp() -> SampleErp:
    return SampleErp()


def test_3way_매칭은_단가_수량_중복_예외를_찾는다(erp: SampleErp) -> None:
    result = {r["invoice"]: r for r in erp.three_way_match("1000")["results"]}
    assert result["INV-V100-7781"]["status"] == "matched"
    assert [i["type"] for i in result["INV-V200-0915"]["issues"]] == ["price_variance"]
    assert result["INV-V200-0915"]["issues"][0]["variance_pct"] == 4.69
    assert [i["type"] for i in result["INV-V300-1102"]["issues"]] == ["quantity_variance"]
    dup = {i["type"]: i for i in result["INV-V200-0917"]["issues"]}
    assert dup["possible_duplicate"]["of_invoice"] == "INV-V200-0915"
    assert dup["quantity_variance"]["over_qty"] == 1_000
    assert erp.three_way_match("2000")["exceptions"] == 0


def test_재주문점_아래_자재와_제안_수량(erp: SampleErp) -> None:
    rows = {r["material"]: r for r in erp.stock_report("1000")["materials"]}
    below = sorted(m for m, r in rows.items() if r["below_reorder_point"])
    assert below == ["M-1001", "M-1003"]
    assert (rows["M-1001"]["available"], rows["M-1001"]["suggested_qty"]) == (60, 150)  # 120 - 140 + 80 open PO
    assert (rows["M-1003"]["available"], rows["M-1003"]["suggested_qty"]) == (520, 500)  # rounded up to MOQ
    erp.create_requisition("1000", lines=[{"material": "M-1001", "qty": 150}])
    assert not erp.stock_report("1000", material="M-1001")["materials"][0]["below_reorder_point"]


def test_채권_연령과_여신_한도(erp: SampleErp) -> None:
    aging = erp.ar_aging("1000")
    assert aging["totals"] == {
        "current": 14_500_000,
        "1-30": 12_000_000,
        "31-60": 9_000_000,
        "61-90": 18_000_000,
        "90+": 3_000_000,
    }
    over = {c["customer"]: c["over_credit_limit"] for c in aging["customers"]}
    assert over == {"C-500": 0, "C-510": 3_000_000, "C-520": 0}


def test_시산표는_균형이고_마감_점검은_남은_항목을_낸다(erp: SampleErp) -> None:
    tb = erp.trial_balance("1000")
    assert tb["balanced"] and tb["total_debit"] == 143_780_000
    close = erp.close_checklist("1000")
    assert close["open_checks"] == ["depreciation", "freight-accrual", "gr-ir-review", "ap-exceptions-blocked"]
    assert erp.close_checklist("2000")["ready_to_close"]


def test_전표_전기는_검증하고_마감_항목을_채운다(erp: SampleErp) -> None:
    with pytest.raises(ErpError, match="다릅니다"):
        erp.post_journal(
            "1000", period=None, text="x", lines=[{"account": "6300", "debit": 1}, {"account": "1590", "credit": 2}]
        )
    with pytest.raises(ErpError, match="열린 회계기간"):
        erp.post_journal(
            "1000",
            period="2026-09",
            text="x",
            lines=[{"account": "6300", "debit": 1}, {"account": "1590", "credit": 1}],
        )
    with pytest.raises(ErpError, match="없는 계정"):
        erp.post_journal(
            "1000", period=None, text="x", lines=[{"account": "9999", "debit": 1}, {"account": "1590", "credit": 1}]
        )
    for check in erp.close_checklist("1000")["checks"]:
        if check.get("proposal"):
            erp.post_journal("1000", period=None, text=check["id"], lines=check["proposal"]["lines"])
    for invoice in ("INV-V200-0915", "INV-V200-0917", "INV-V300-1102"):
        erp.set_payment_block("1000", invoice=invoice, block=True, reason="match exception")
    assert erp.close_checklist("1000")["open_checks"] == ["gr-ir-review"]
    assert erp.trial_balance("1000")["balanced"]


def test_회사_코드가_데이터를_나누고_라우트는_거부를_도구_오류로_낸다(erp: SampleErp) -> None:
    routes = erp_routes(erp)
    assert routes["erp-stock-read"](None, {"company": "2000"})["materials"][0]["material"] == "M-9001"
    with pytest.raises(ToolException, match="없는 회사 코드"):
        routes["erp-stock-read"](None, {"company": "9999"})
    with pytest.raises(ToolException, match="없는 매입송장"):
        routes["erp-payment-block"](None, {"company": "2000", "invoice": "INV-V200-0915", "reason": "r"})
