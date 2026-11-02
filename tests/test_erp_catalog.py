"""The shipped catalog (catalog/): ERP agents and actions on the sample ERP."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage, ToolMessage

from tests.conftest import SHIPPED_CATALOG_DIR, scripted_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import AssetKind
from wigtn_agents.context import default_context
from wigtn_agents.middleware.scope_guard import SCOPE_PARAMS
from wigtn_agents.runner import run_local
from wigtn_agents.sample_erp import shared_erp
from wigtn_agents.testing import ScriptedFakeModel

AGENTS = ["erp-assistant", "ap-clerk", "inventory-planner", "close-accountant", "router"]
ACTIONS = ["close-readiness", "collections-priority", "invoice-match-review", "replenishment-proposal"]


@pytest.fixture
def erp_catalog(monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("WIGTN_AGENT_CATALOG_DIR", str(SHIPPED_CATALOG_DIR))
    return Catalog.load(SHIPPED_CATALOG_DIR / "assets", SHIPPED_CATALOG_DIR / "manifests")


def test_카탈로그에는_ERP_에이전트와_action이_있다(erp_catalog: Catalog) -> None:
    assert [erp_catalog.manifest(a).agent_id for a in AGENTS] == AGENTS
    assert sorted(erp_catalog.action_ids()) == ACTIONS


def test_회사_코드를_받는_ERP_도구는_모두_스코프_가드에_등록돼_있다() -> None:
    for path in (SHIPPED_CATALOG_DIR / "assets").glob("tool.erp-*.yaml"):
        asset = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "company" in asset["body"]["input_schema"]["properties"], path.name
        assert SCOPE_PARAMS.get(asset["name"]) == (("company", "entity"),), path.name


def test_쓰기_도구는_모두_승인_지점_뒤에_있다(erp_catalog: Catalog) -> None:
    for agent_id in AGENTS:
        manifest = erp_catalog.manifest(agent_id)
        for tool in erp_catalog.tools_for_manifest(manifest):
            if tool.typed_body.side_effect == "write":  # type: ignore[attr-defined]
                assert f"tool://{tool.name}@active" in manifest.interrupts.approve_before, (agent_id, tool.name)


def test_llm_채점_케이스는_모두_루브릭이_있고_스킬마다_케이스가_있다(erp_catalog: Catalog) -> None:
    for path in (SHIPPED_CATALOG_DIR / "assets").glob("skill.*.yaml"):
        asset = erp_catalog.resolve(f"skill://{yaml.safe_load(path.read_text())['name']}@active")
        assert asset.kind == AssetKind.SKILL and asset.eval_cases, path.name
        assert all(c.rubric for c in asset.eval_cases if c.judge == "llm")


async def test_결산_에이전트는_승인_후에만_전표를_전기한다(erp_catalog: Catalog) -> None:
    ctx = default_context("close-accountant", entity="1000", principal="u-1")
    stopped = await run_local(
        "close-accountant", "10월 감가상각 전표 올려줘", ctx, catalog=erp_catalog, model=ScriptedFakeModel(), sink=[]
    )
    assert stopped.status == "interrupted" and stopped.pending[0]["name"] == "erp-journal-post"
    assert "depreciation" in shared_erp().close_checklist("1000")["open_checks"]

    done = await run_local(
        "close-accountant",
        "10월 감가상각 전표 올려줘",
        ctx,
        catalog=erp_catalog,
        model=ScriptedFakeModel(),
        sink=[],
        approve=True,
    )
    assert done.status == "success" and "JE-1000-2026-10-004" in done.answer
    assert "depreciation" not in shared_erp().close_checklist("1000")["open_checks"]


async def test_모델이_다른_회사_코드를_보내도_run의_회사로_조회한다(erp_catalog: Catalog) -> None:
    ctx = default_context("erp-assistant", entity="2000", principal="u-1")
    model = scripted_model(
        AIMessage(content="", tool_calls=[{"name": "erp-stock-read", "args": {"company": "1000"}, "id": "c1"}]),
        AIMessage(content="끝"),
    )
    await run_local("erp-assistant", "재고 봐줘", ctx, catalog=erp_catalog, model=model, sink=[])
    tool_reply = next(m for m in model.seen_messages[-1] if isinstance(m, ToolMessage))
    assert '"company": "2000"' in tool_reply.text and "M-9001" in tool_reply.text and "M-1001" not in tool_reply.text


@pytest.mark.parametrize("action_id", ACTIONS)
async def test_action은_모두_가짜_모델로_끝까지_돈다(
    erp_catalog: Catalog, action_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIGTN_AGENT_ACTION_LOG", str(tmp_path / "actions.jsonl"))
    ctx = default_context(action_id, entity="1000", principal="u-1")
    result = await run_local(action_id, {}, ctx, catalog=erp_catalog, model=ScriptedFakeModel(), sink=[])
    assert result.kind == "action" and result.status == "success", result


def test_스킬_본문의_라우팅_표도_프롬프트에_들어간다(erp_catalog: Catalog) -> None:
    from wigtn_agents.middleware.asset_loader import compose_prompt, pin_assets

    manifest = erp_catalog.manifest("router")
    prompt = compose_prompt(manifest, erp_catalog, pin_assets(manifest, erp_catalog))
    assert all(
        f"agent_id: {a}" in prompt for a in ("ap-clerk", "inventory-planner", "close-accountant", "erp-assistant")
    )
