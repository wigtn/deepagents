"""Manifest subagents: own tools, own gates, the parent's scope, read-only, one level."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from tests.conftest import scripted_model, user_turn
from wigtn_agents.assembly import GENERAL_PURPOSE_NAME, assemble, build_subagents, resolve_run_model
from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.context import default_context
from wigtn_agents.manifest import Manifest
from wigtn_agents.tools.gateway import ToolClients, ToolGatewayClient


def test_서브에이전트는_자기_도구와_게이트를_갖고_general_purpose가_마지막이다(catalog: Catalog) -> None:
    ctx = default_context("records-writer", entity="1000")
    manifest = catalog.manifest("records-writer")
    specs = build_subagents(manifest, catalog, ctx, sink=[], model=resolve_run_model(manifest, ctx))

    assert [s["name"] for s in specs] == ["records-guide", GENERAL_PURPOSE_NAME]
    guide = specs[0]
    assert sorted(t.name for t in guide["tools"]) == ["record-read", "view-read"]  # type: ignore[union-attr]
    names = [type(m).__name__ for m in guide["middleware"]]
    assert names.index("EventRecorderMiddleware") < names.index("PermissionFilterMiddleware")
    assert "ScopeGuardMiddleware" in names
    assert "스킬 · example/checklist" in guide["system_prompt"]
    assert "model" not in guide  # the parent's model: one backbone per run


def test_서브에이전트의_스코프는_부모_run_context에서_온다(catalog: Catalog) -> None:
    # Given: the parent delegates; the subagent's model calls record-read with a forged entity
    ctx = default_context("records-writer", principal="u-1", entity="1000")
    gateway = ToolGatewayClient()
    events: list[dict[str, Any]] = []
    parent = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "8월 미처리 조회", "subagent_type": "records-guide"},
                    "id": "t1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "record-read", "args": {"period": "2026-08", "entity": "9999"}, "id": "r1"}],
        ),
        AIMessage(content="미처리 2건"),
        AIMessage(content="위임 결과: 미처리 2건"),
    )
    agent = assemble(
        catalog.manifest("records-writer"),
        catalog,
        ctx,
        model=parent,
        clients=ToolClients(gateway=gateway),
        sink=events,
    )

    # When
    agent.invoke(user_turn("8월 점검"), context=ctx)

    # Then: the gateway saw the canonical entity, and the subagent's rows are stamped as such
    assert [c["payload"]["entity"] for c in gateway.calls] == ["1000"]
    sub_rows = [e for e in events if e["role"] == "subagent"]
    assert sub_rows and {e["agent_id"] for e in sub_rows} == {"records-guide"}
    assert all(e["asset_versions"] for e in sub_rows)


def _with_subagent(catalog: Catalog, sub: str) -> Manifest:
    return catalog.manifest("records-guide").model_copy(update={"subagents": [sub]})


def test_쓰기_도구가_있는_서브에이전트는_조립하지_않는다(catalog: Catalog) -> None:
    ctx = default_context("records-guide", entity="1000")
    writer = catalog.manifest("records-writer")
    catalog._manifests["records-writer"] = writer.model_copy(update={"subagents": []})
    with pytest.raises(ValueError, match="읽기 전용"):
        assemble(_with_subagent(catalog, "records-writer"), catalog, ctx, model=scripted_model(), sink=[])


def test_서브에이전트는_한_단계만_둔다(catalog: Catalog) -> None:
    nested = catalog.manifest("records-guide").model_copy(update={"subagents": ["router"]})
    catalog._manifests["records-guide"] = nested
    ctx = default_context("router")
    manifest = catalog.manifest("router").model_copy(update={"subagents": ["records-guide"]})
    with pytest.raises(ValueError, match="한 단계"):
        assemble(manifest, catalog, ctx, model=scripted_model(), sink=[])
