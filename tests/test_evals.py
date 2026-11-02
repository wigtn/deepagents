"""Eval judges: rule checks, rubric scoring and the judge-is-not-the-backbone rule."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from wigtn_agents.assets.catalog import Catalog
from wigtn_agents.assets.schema import EvalCase
from wigtn_agents.evals import JudgeVerdict, llm_grade, resolve_judge, rubric_hash, rule_grade
from wigtn_agents.models import ModelRegistry

RUBRIC = {
    "pass_score": 0.75,
    "criteria": [
        {"id": "scope", "description": "states scope", "weight": 2},
        {"id": "approval", "description": "asks approval", "weight": 2},
        {"id": "honest", "description": "no false claim", "weight": 1},
    ],
}


def _case(**over: Any) -> EvalCase:
    return EvalCase.model_validate({"id": "c", "input": {"question": "q"}, "judge": "llm", "rubric": RUBRIC, **over})


class _Judge:
    """Chat-model stand-in whose structured output is a fixed verdict."""

    def __init__(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        self.seen: list[Any] = []

    def with_structured_output(self, schema: Any, **_: Any) -> Any:
        def produce(messages: Any) -> dict[str, Any]:
            self.seen.append(messages)
            raw = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            return {"raw": raw, "parsed": schema.model_validate(self.verdict), "parsing_error": None}

        return RunnableLambda(produce)


def _registry(**over: Any) -> ModelRegistry:
    return ModelRegistry.model_validate(
        {
            "default": "models/a",
            "models": {
                "models/a": {"provider": "openai", "model": "m-1", "price": {"input": 1, "output": 1}},
                "models/a-again": {"provider": "openai", "model": "m-1", "price": {"input": 1, "output": 1}},
                "models/b": {"provider": "anthropic", "model": "m-2", "price": {"input": 1, "output": 1}},
            },
            **over,
        }
    )


def test_채점_방식마다_필요한_값이_없으면_케이스가_만들어지지_않는다() -> None:
    with pytest.raises(ValidationError, match="rubric"):
        EvalCase.model_validate({"id": "x", "input": {}, "judge": "llm"})
    with pytest.raises(ValidationError, match="expected.text"):
        EvalCase.model_validate({"id": "x", "input": {}, "judge": "exact"})
    with pytest.raises(ValidationError, match="expected.contains"):
        EvalCase.model_validate({"id": "x", "input": {}, "judge": "contains"})
    with pytest.raises(ValidationError, match="겹칩니다"):
        _case(rubric={"criteria": [{"id": "a", "description": "d"}, {"id": "a", "description": "e"}]})


def test_contains와_exact() -> None:
    contains = EvalCase.model_validate({"id": "x", "input": {}, "expected": {"contains": ["기간", "범위"]}})
    assert rule_grade(contains, "기간과 범위를 확인").verdict == "passed"
    assert rule_grade(contains, "기간만").verdict == "failed"
    exact = EvalCase.model_validate({"id": "x", "input": {}, "judge": "exact", "expected": {"text": "ok"}})
    assert rule_grade(exact, " ok\n").verdict == "passed"


async def test_점수는_채점_모델이_아니라_가중치로_계산한다() -> None:
    judge_model = _Judge(
        {
            "criteria": [
                {"id": "scope", "met": True, "reason": "r"},
                {"id": "approval", "met": True, "reason": "r"},
                {"id": "honest", "met": False, "reason": "r"},
            ]
        }
    )
    reg = _registry()
    grade = await llm_grade(_case(), "answer", judge_model=judge_model, judge=reg.resolve("models/b"))
    assert grade.verdict == "passed" and grade.score == 0.8 and grade.pass_score == 0.75
    assert grade.judge_model_id == "anthropic:m-2" and grade.rubric_hash == rubric_hash(_case().rubric)  # type: ignore[arg-type]
    assert grade.judge_usage == {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0, "reasoning_tokens": 0}
    prompt = judge_model.seen[0][1].content
    assert "Answer to grade:\nanswer" in prompt and "- approval: asks approval" in prompt


async def test_채점_모델이_빠뜨린_기준은_충족하지_않은_것으로_본다() -> None:
    judge_model = _Judge(
        {"criteria": [{"id": "scope", "met": True, "reason": "r"}, {"id": "bogus", "met": True, "reason": "r"}]}
    )
    grade = await llm_grade(_case(), "a", judge_model=judge_model, judge=_registry().resolve("models/b"))
    assert grade.verdict == "failed" and grade.score == 0.4 and grade.missing == ["approval", "honest"]


def test_채점_모델은_평가_대상과_다른_모델이어야_한다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIGTN_AGENT_FAKE_MODEL", raising=False)
    reg = _registry()
    backbone = reg.resolve("models/a")
    with pytest.raises(ValueError, match="같은 모델"):
        resolve_judge(reg, backbone, "models/a")
    with pytest.raises(ValueError, match="같은 모델"):
        resolve_judge(reg, backbone, "models/a-again")  # another alias of the same model
    with pytest.raises(LookupError, match="채점 모델이 없습니다"):
        resolve_judge(reg, backbone)
    assert resolve_judge(reg, backbone, "models/b").alias == "models/b"
    assert resolve_judge(_registry(judge="models/b"), backbone).alias == "models/b"
    monkeypatch.setenv("WIGTN_AGENT_FAKE_MODEL", "1")
    with pytest.raises(ValueError, match="--no-llm"):
        resolve_judge(reg, backbone, "models/b")


def test_registry의_judge도_등록된_별칭이어야_한다() -> None:
    with pytest.raises(ValidationError, match="judge"):
        _registry(judge="models/none")


def test_카탈로그의_llm_케이스는_루브릭을_갖는다(catalog: Catalog) -> None:
    cases = catalog.eval_cases_for(catalog.manifest("records-guide").all_asset_refs())
    llm = [c for _, c in cases if c.judge == "llm"]
    assert llm and all(c.rubric for c in llm)
    assert JudgeVerdict.model_json_schema()["required"] == ["criteria"]
