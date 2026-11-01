"""Asset protocol, manifest and catalog (ported from the reference design samples)."""

from __future__ import annotations

import pytest
import yaml

from tests.conftest import CATALOG_DIR
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import Asset
from wigtn_agents.manifest import FlowTemplate, Manifest, decide_flow


def test_모든_자산_yaml이_스키마를_통과한다() -> None:
    files = sorted((CATALOG_DIR / "assets").glob("*.yaml"))
    assert len(files) == 10
    for path in files:
        Asset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_모든_매니페스트가_스키마를_통과한다() -> None:
    files = sorted((CATALOG_DIR / "manifests").glob("*.yaml"))
    assert len(files) == 4
    for path in files:
        Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


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
