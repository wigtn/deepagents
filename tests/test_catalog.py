"""Asset protocol, manifest and catalog (ported from the reference design samples)."""

from __future__ import annotations

import pytest
import yaml

from tests.conftest import CATALOG_DIR
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.manifest import ActionManifest, FlowTemplate, Manifest, decide_flow


def test_모든_자산_yaml이_스키마를_통과한다() -> None:
    files = sorted((CATALOG_DIR / "assets").glob("*.yaml"))
    assert len(files) == 10
    for path in files:
        Asset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_모든_매니페스트가_스키마를_통과한다() -> None:
    files = sorted((CATALOG_DIR / "manifests").glob("*.yaml"))
    assert len(files) == 6
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        (ActionManifest if data["kind"] == "action" else Manifest).model_validate(data)


def test_카탈로그는_에이전트와_action을_나눠_싣는다(catalog: Catalog) -> None:
    assert catalog.agent_ids() == ["custom-flow", "records-guide", "records-writer", "router"]
    assert catalog.action_ids() == ["records-review", "records-summary"]
    with pytest.raises(KeyError, match="등록되지 않은 에이전트"):
        catalog.manifest("records-summary")


def test_등록되지_않은_서브에이전트는_로드에서_실패한다(catalog: Catalog) -> None:
    data = catalog.manifest("records-writer").model_dump(mode="json", by_alias=True)
    data["metadata"]["name"] = "broken"
    data["subagents"] = ["nobody"]
    catalog.add_manifest(Manifest.model_validate(data))
    with pytest.raises(ValueError, match="nobody"):
        catalog.validate_links()


def test_action_output_schema는_object여야_한다() -> None:
    with pytest.raises(ValueError, match="output_schema"):
        ActionManifest.model_validate(
            {
                "apiVersion": "wigtn.agents/v1",
                "kind": "action",
                "metadata": {"name": "x", "version": 1, "owner": "t"},
                "level": "structured",
                "skill": "skill://example/checklist@active",
                "output_schema": {"type": "string"},
            }
        )


def test_mutable_by_sel에_permission_scope는_들어갈_수_없다(catalog: Catalog) -> None:
    data = catalog.resolve("skill://example/checklist@active").model_dump(mode="json")
    data["mutable_by_sel"] = ["permission_scope"]
    with pytest.raises(ValueError):
        Asset.model_validate(data)


def test_카탈로그가_active를_정수_버전으로_고정한다(catalog: Catalog) -> None:
    assert catalog.pin("skill://example/checklist@active") == "skill://example/checklist@3"
    assert catalog.resolve("tool://record-write@active").typed_body.side_effect == "write"  # type: ignore[attr-defined]


def test_flow_판단표() -> None:
    assert decide_flow(has_write_tool=False, multi_step=False, needs_intermediate_state=False) == FlowTemplate.REACT
    assert (
        decide_flow(has_write_tool=False, multi_step=True, needs_intermediate_state=True) == FlowTemplate.PLAN_EXECUTE
    )
    assert (
        decide_flow(has_write_tool=True, multi_step=True, needs_intermediate_state=True) == FlowTemplate.APPROVAL_GATED
    )
    assert (
        decide_flow(has_write_tool=True, multi_step=True, needs_intermediate_state=True, expressible=False)
        == FlowTemplate.EXTENSION
    )


def test_tools_for_manifest는_스킬이_참조한_도구만_고정_버전으로_모은다(catalog: Catalog) -> None:
    names = sorted(a.tool_name() for a in catalog.tools_for_manifest(catalog.manifest("records-writer")))
    assert names == ["record-read", "record-write", "view-read"]
