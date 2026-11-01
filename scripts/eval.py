"""Run the eval cases an agent's or action's assets carry, in-process, and write a report.

    PYTHONPATH=src uv run python scripts/eval.py records-guide
    PYTHONPATH=src uv run python scripts/eval.py records-summary --model models/qwen3.8-flash --batch backbone-qwen
    PYTHONPATH=src uv run python scripts/eval.py records-guide --model models/fake      # offline

Each skill and prompt asset can carry ``eval_cases`` (input, expected, judge). The cases of every
asset the manifest reaches run once each with ``session_type=evaluation`` and one ``batch_id``,
so the event log rows of this eval can be selected later. Judges: ``contains`` (every expected
string appears), ``exact``; ``llm`` cases are reported as skipped.

Output: ``var/eval/<agent>-<batch>.jsonl`` (one row per case) and a summary line. Exit code 1 if
any case failed.
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
from wigtn_agents.runner import run_local  # noqa: E402
from wigtn_agents.worker import load_catalog  # noqa: E402


def judge(case: EvalCase, text: str) -> str:
    """passed | failed | skipped."""
    if case.judge == "contains":
        return "passed" if all(str(s) in text for s in case.expected.get("contains", [])) else "failed"
    if case.judge == "exact":
        return "passed" if text.strip() == str(case.expected.get("text", "")).strip() else "failed"
    return "skipped"


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
    args = parser.parse_args()

    catalog = load_catalog()
    is_action = args.agent_id in catalog.action_ids()
    manifest = catalog.action(args.agent_id) if is_action else catalog.manifest(args.agent_id)
    batch = args.batch or f"eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    out = ROOT / "var" / "eval" / f"{args.agent_id}-{batch}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = {"passed": 0, "failed": 0, "skipped": 0}
    with out.open("w", encoding="utf-8") as report:
        for ref, case in catalog.eval_cases_for(manifest.all_asset_refs()):
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
            if case.judge == "llm":
                verdict, text, status = "skipped", "", "not-run"
            else:
                result = await run_local(args.agent_id, payload, ctx, catalog=catalog)
                text, status = result.text(), result.status
                verdict = judge(case, text)
            counts[verdict] += 1
            row = {
                "batch_id": batch,
                "agent_id": args.agent_id,
                "asset": ref,
                "case": case.id,
                "judge": case.judge,
                "verdict": verdict,
                "status": status,
                "model": args.model or manifest.model.ref,
                "output": text[:2000],
            }
            report.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{verdict:7s} {ref} {case.id}")

    print(json.dumps({"batch_id": batch, "report": str(out), **counts}, ensure_ascii=False))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
