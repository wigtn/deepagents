"""Judges for the eval cases assets carry: ``contains``, ``exact`` and a rubric-based ``llm`` judge.

The ``llm`` judge never grades its own answers: the judge model must be another provider model
than the backbone under evaluation (two aliases of the same model count as the same). The judge
only says, per rubric criterion, whether the answer meets it and why; the score is computed here
(weights of met criteria / all weights) and compared with the rubric's ``pass_score``. A criterion
the judge leaves out counts as not met and is listed in ``missing``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from wigtn_agents.assets.schema import EvalCase, Rubric
from wigtn_agents.middleware.events import usage_of
from wigtn_agents.models import ModelRegistry, ResolvedModel, fake_forced

Verdict = Literal["passed", "failed", "skipped"]

JUDGE_SYSTEM = """You grade one answer of an AI agent against a rubric.
For every criterion, decide whether the answer meets it and give a one-sentence reason.
Judge only what the answer itself says. Do not reward claims of actions the answer cannot show.
A reference answer, when given, illustrates a good answer; it is not the only correct wording.
Return every criterion id exactly once."""


class CriterionVerdict(BaseModel):
    """The judge's call on one criterion."""

    id: str
    met: bool
    reason: str = Field(description="one sentence")


class JudgeVerdict(BaseModel):
    """What the judge model returns."""

    criteria: list[CriterionVerdict]


class Grade(BaseModel):
    """Outcome of one case."""

    verdict: Verdict
    score: Optional[float] = None
    pass_score: Optional[float] = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    judge_alias: Optional[str] = None
    judge_model_id: Optional[str] = None
    rubric_hash: Optional[str] = None
    judge_usage: Optional[dict[str, int]] = None


def rule_grade(case: EvalCase, text: str) -> Grade:
    """``contains`` (every expected string appears) and ``exact`` (whitespace-trimmed equality)."""
    if case.judge == "contains":
        ok = all(str(s) in text for s in case.expected["contains"])
    elif case.judge == "exact":
        ok = text.strip() == str(case.expected["text"]).strip()
    else:
        raise ValueError(f"{case.id}: llm 채점은 llm_grade로 합니다")
    return Grade(verdict="passed" if ok else "failed")


def rubric_hash(rubric: Rubric) -> str:
    """Stable short hash of a rubric, so reports say which rubric graded them."""
    blob = json.dumps(rubric.model_dump(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def same_model(a: ResolvedModel, b: ResolvedModel) -> bool:
    """Same provider model, whatever the alias."""
    return a.model_id == b.model_id


def resolve_judge(registry: ModelRegistry, backbone: ResolvedModel, alias: Optional[str] = None) -> ResolvedModel:
    """The judge model for an eval of ``backbone``.

    Raises:
        LookupError: no judge given and none set in the registry.
        ValueError: the judge is the backbone, or the fake model replaces every model.

    """
    if fake_forced():
        raise ValueError(
            "WIGTN_AGENT_FAKE_MODEL이 켜져 있으면 채점 모델도 가짜가 됩니다. llm 케이스는 --no-llm으로 빼세요"
        )
    name = alias or registry.judge
    if name is None:
        raise LookupError("채점 모델이 없습니다. --judge-model을 주거나 catalog/models.yaml에 judge를 적으세요")
    judge = registry.resolve(name)
    if same_model(judge, backbone):
        raise ValueError(f"채점 모델 {judge.alias}가 평가 대상 {backbone.alias}와 같은 모델({judge.model_id})입니다")
    return judge


def judge_messages(case: EvalCase, answer: str) -> list[Any]:
    """The judge's prompt: the task input, the optional reference, the answer and the criteria."""
    assert case.rubric is not None
    criteria = "\n".join(f"- {c.id}: {c.description}" for c in case.rubric.criteria)
    parts = [f"Task input:\n{json.dumps(case.input, ensure_ascii=False)}"]
    reference = case.expected.get("reference")
    if reference:
        parts.append(f"Reference answer:\n{reference}")
    parts.append(f"Answer to grade:\n{answer}")
    parts.append(f"Criteria:\n{criteria}")
    return [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content="\n\n".join(parts))]


def score_verdict(rubric: Rubric, verdict: JudgeVerdict) -> tuple[float, list[dict[str, Any]], list[str]]:
    """Weighted share of met criteria; unknown ids ignored, absent ids count as not met."""
    calls = {c.id: c for c in verdict.criteria}
    total = sum(c.weight for c in rubric.criteria)
    met = 0.0
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for criterion in rubric.criteria:
        call = calls.get(criterion.id)
        if call is None:
            missing.append(criterion.id)
            rows.append({"id": criterion.id, "met": False, "reason": None, "weight": criterion.weight})
            continue
        met += criterion.weight if call.met else 0.0
        rows.append({"id": criterion.id, "met": call.met, "reason": call.reason, "weight": criterion.weight})
    return round(met / total, 4), rows, missing


async def llm_grade(case: EvalCase, answer: str, *, judge_model: Any, judge: ResolvedModel) -> Grade:
    """Grade one answer with the rubric. ``judge_model`` is a chat model (built from ``judge``)."""
    rubric = case.rubric
    if rubric is None:
        raise ValueError(f"{case.id}: rubric이 없습니다")
    out = await judge_model.with_structured_output(JudgeVerdict, include_raw=True).ainvoke(judge_messages(case, answer))
    parsed = out.get("parsed") if isinstance(out, dict) else out
    if not isinstance(parsed, JudgeVerdict):
        parsed = JudgeVerdict.model_validate(parsed or {"criteria": []})
    score, rows, missing = score_verdict(rubric, parsed)
    return Grade(
        verdict="passed" if score >= rubric.pass_score else "failed",
        score=score,
        pass_score=rubric.pass_score,
        criteria=rows,
        missing=missing,
        judge_alias=judge.alias,
        judge_model_id=judge.model_id,
        rubric_hash=rubric_hash(rubric),
        judge_usage=usage_of(out.get("raw")) if isinstance(out, dict) else None,
    )
