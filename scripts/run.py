"""Run one agent or action — through Aegra (default) or in-process (``--local``).

    PYTHONPATH=src uv run python scripts/run.py records-guide "8월 기록 점검 부탁해"
    PYTHONPATH=src uv run python scripts/run.py records-summary '{"period": "2026-08", "view": "open-items"}'
    PYTHONPATH=src uv run python scripts/run.py records-guide "점검" --local --model models/fake
    PYTHONPATH=src uv run python scripts/run.py records-writer "8월 기록 반영" --batch exp-1 --middleware my-defense

Whether the name is an agent or an action comes from the catalog. An agent takes text, an action
takes a JSON object. ``--model`` and ``--middleware`` select from what the image already has
(``catalog/models.yaml``, registered middleware names). The run record goes to the event log
(``WIGTN_AGENT_EVENT_LOG``, default ``var/events.jsonl``) with the same keys either way.
Through Aegra, run ``scripts/register_assistants.py`` once after adding a manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

from wigtn_agents.context import default_context  # noqa: E402
from wigtn_agents.worker import load_catalog  # noqa: E402


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent_id")
    parser.add_argument("input", help="user text (agent) or a JSON object (action)")
    parser.add_argument("--local", action="store_true", help="run in-process instead of through Aegra")
    parser.add_argument("--model", help="model alias from catalog/models.yaml")
    parser.add_argument("--middleware", action="append", default=[], help="registered extra middleware (repeatable)")
    parser.add_argument("--batch", help="batch_id stamped on every event")
    parser.add_argument("--entity", default="1000", help="scope value the scope guard injects")
    parser.add_argument("--session-type", default="test", choices=["production", "evaluation", "test"])
    parser.add_argument("--approve", action="store_true", help="--local: approve every pending approval")
    return parser.parse_args()


def payload_for(kind: str, raw: str) -> Any:
    if kind == "agent":
        return raw
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit("action input must be a JSON object")
    return value


def configurable(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "agent_id": args.agent_id,
        "session_type": args.session_type,
        "entity": args.entity,
        "batch_id": args.batch,
        "model": args.model,
        "middleware": args.middleware,
    }
    return {k: v for k, v in values.items() if v not in (None, [])}


async def run_local(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    from wigtn_agents.runner import run_local as run

    catalog = load_catalog()
    ctx = default_context(
        args.agent_id, principal="local", **{k: v for k, v in configurable(args).items() if k != "agent_id"}
    )
    result = await run(args.agent_id, payload_for(kind, args.input), ctx, catalog=catalog, approve=args.approve)
    return {
        "agent_id": result.agent_id,
        "kind": result.kind,
        "thread_id": result.thread_id,
        "status": result.status,
        "answer": result.answer if kind == "agent" else None,
        "result": result.result,
        "pending": result.pending,
    }


async def run_remote(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    from e2e_common import assistant_id_for, client, last_ai_text

    c = client()
    thread = await c.threads.create()
    tid = thread["thread_id"]
    run_input = (
        {"messages": [{"role": "user", "content": args.input}]}
        if kind == "agent"
        else {"input": payload_for(kind, args.input)}
    )
    # The full configurable every time: Aegra merges assistant and run config shallowly.
    final = await c.runs.wait(
        tid, assistant_id_for(args.agent_id), input=run_input, config={"configurable": configurable(args)}
    )
    runs = await c.runs.list(tid, limit=1)
    values = final if isinstance(final, dict) else {}
    return {
        "agent_id": args.agent_id,
        "kind": kind,
        "thread_id": tid,
        "run_id": runs[0]["run_id"] if runs else None,
        "status": runs[0]["status"] if runs else None,
        "answer": last_ai_text(values.get("messages", [])) if kind == "agent" else None,
        "result": values.get("result") if kind == "action" else None,
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse()
    catalog = load_catalog()
    if args.agent_id in catalog.action_ids():
        kind = "action"
    elif catalog.has_agent(args.agent_id):
        kind = "agent"
    else:
        raise SystemExit(f"unknown agent or action: {args.agent_id}")
    outcome = asyncio.run(run_local(args, kind) if args.local else run_remote(args, kind))
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
