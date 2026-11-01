"""In-process runs used by scripts/run.py --local and scripts/eval.py."""

from __future__ import annotations

from typing import Any

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import default_context
from wigtn_agents.runner import run_local
from wigtn_agents.testing import ScriptedFakeModel


async def test_승인_지점은_멈추고_approve면_이어서_끝난다(catalog: Catalog) -> None:
    ctx = default_context("records-writer", entity="1000")
    stopped = await run_local(
        "records-writer", "8월 기록 반영", ctx, catalog=catalog, model=ScriptedFakeModel(), sink=[]
    )
    assert stopped.status == "interrupted" and stopped.pending[0]["name"] == "record-write"

    events: list[dict[str, Any]] = []
    done = await run_local(
        "records-writer", "8월 기록 반영", ctx, catalog=catalog, model=ScriptedFakeModel(), sink=events, approve=True
    )
    assert done.status == "success" and "반영 완료" in done.answer
    assert [e["status"] for e in events if e["event_type"] == "run.end"] == ["interrupted", "ok"]


async def test_action은_입력_dict로_돌고_결과를_돌려준다(catalog: Catalog) -> None:
    ctx = default_context("records-review", entity="1000")
    result = await run_local(
        "records-review", {"period": "2026-08"}, ctx, catalog=catalog, model=ScriptedFakeModel(), sink=[]
    )
    assert result.kind == "action" and result.result == {"verdict": "(fake)", "reasons": []}
    assert '"verdict"' in result.text()
