"""Run the eval cases an agent's or action's assets carry, in-process, and write a report.

    PYTHONPATH=src uv run python scripts/eval.py records-guide
    PYTHONPATH=src uv run python scripts/eval.py records-summary --model models/qwen3.8-flash --batch backbone-qwen
    PYTHONPATH=src uv run python scripts/eval.py records-guide --judge-model models/<other>
    PYTHONPATH=src uv run python scripts/eval.py records-guide --model models/fake --no-llm   # offline

Each skill and prompt asset can carry ``eval_cases`` (input, expected, judge). The cases of every
asset the manifest reaches run once each with ``session_type=evaluation`` and one ``batch_id``,
so the event log rows of this eval can be selected later. Judges:

* ``contains``: every string of ``expected.contains`` appears in the answer;
* ``exact``: the answer equals ``expected.text``;
* ``llm``: a judge model grades the answer against the case's ``rubric`` (criteria with weights
  and a ``pass_score``). The judge must be another model than the backbone; the eval stops before
  the first run if it is the same. ``--no-llm`` reports these cases as skipped instead.

Output: ``var/eval/<agent>-<batch>.jsonl`` (one row per case, with per-criterion verdicts, score,
judge model and rubric hash for ``llm`` cases) and a summary line. Exit code 1 if any case
failed, 2 if the judge cannot be used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from wigtn_agents.assets.schema import EvalCase  # noqa: E402
from wigtn_agents.context import default_context  # noqa: E402
from wigtn_agents.evals import Grade, llm_grade, resolve_judge, rule_grade  # noqa: E402
from wigtn_agents.models import build_model, load_registry  # noqa: E402
from wigtn_agents.runner import run_local  # noqa: E402
from wigtn_agents.worker import load_catalog  # noqa: E402


def user_text(case: EvalCase) -> str:
    if "question" in case.input:
        return str(case.input["question"])
    return json.dumps(case.input, ensure_ascii=False)


async def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent_id")
    parser.add_argument("--model", help="model alias from catalog/models.yaml")
    parser.add_argument("--middleware", action="append", default=[])
    parser.add_argument("--batch", help="batch_id (default eval-<utc timestamp>)")
    parser.add_argument("--entity", default="1000", help="scope when a case input has no entity")
    parser.add_argument("--judge-model", help="judge alias for llm cases (default: judge in catalog/models.yaml)")
    parser.add_argument("--no-llm", action="store_true", help="report llm cases as skipped")
    args = parser.parse_args()

    catalog = load_catalog()
    is_action = args.agent_id in catalog.action_ids()
    manifest = catalog.action(args.agent_id) if is_action else catalog.manifest(args.agent_id)
    batch = args.batch or f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    cases = catalog.eval_cases_for(manifest.all_asset_refs())

    registry = load_registry()
    backbone = registry.resolve(args.model or manifest.model.ref)
    judge = judge_model = None
    if not args.no_llm and any(c.judge == "llm" for _, c in cases):
        try:
            judge = resolve_judge(registry, backbone, args.judge_model)
            judge_model = build_model(judge.alias, registry=registry)
        except (LookupError, ValueError, KeyError) as exc:
            print(f"채점 모델을 쓸 수 없습니다: {exc}", file=sys.stderr)
            return 2

    out = ROOT / "var" / "eval" / f"{args.agent_id}-{batch}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = {"passed": 0, "failed": 0, "skipped": 0}
    with out.open("w", encoding="utf-8") as report:
        for ref, case in cases:
            ctx = default_context(
                args.agent_id,
                principal="eval",
                session_type="evaluation",
                batch_id=batch,
                entity=str(case.input.get("entity") or args.entity),
                model=args.model,
                middleware=args.middleware,
            )
            payload: Any = dict(case.input) if is_action else user_text(case)
            if case.judge == "llm" and judge is None:
                grade, text, status = Grade(verdict="skipped"), "", "not-run"
            else:
                result = await run_local(args.agent_id, payload, ctx, catalog=catalog)
                text, status = result.text(), result.status
                if case.judge == "llm":
                    grade = await llm_grade(case, text, judge_model=judge_model, judge=judge)  # type: ignore[arg-type]
                else:
                    grade = rule_grade(case, text)
            verdict = grade.verdict
            counts[verdict] += 1
            row = {
                "batch_id": batch,
                "agent_id": args.agent_id,
                "asset": ref,
                "case": case.id,
                "judge": case.judge,
                "verdict": verdict,
                "status": status,
                "model": backbone.alias,
                "model_id": backbone.model_id,
                **grade.model_dump(exclude={"verdict"}, exclude_none=True),
                "output": text[:2000],
            }
            report.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{verdict:7s} {ref} {case.id}")

    print(json.dumps({"batch_id": batch, "report": str(out), **counts}, ensure_ascii=False))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
