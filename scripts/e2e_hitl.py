"""e2e (b)(c): approval-gated run on close-accountant.

``start``  — run until the erp-journal-post interrupt; saves thread/run ids to var/e2e_state.json.
``resume`` — (optionally after restarting the server) approve and finish on the same thread.
"""

from __future__ import annotations

import asyncio
import sys

from e2e_common import (
    assistant_id_for,
    check,
    client,
    events_from_log,
    last_ai_text,
    load_state,
    run_configurable,
    save_state,
    wait_run,
)

AGENT_ID = "close-accountant"
WRITE = "erp-journal-post"


async def start() -> None:
    """Run until the interrupt."""
    c = client()
    thread = await c.threads.create()
    tid = thread["thread_id"]
    run = await c.runs.create(
        tid,
        assistant_id_for(AGENT_ID),
        input={"messages": [{"role": "user", "content": "10월 감가상각 전표 올려줘"}]},
        config={"configurable": run_configurable(AGENT_ID)},
    )
    run = await wait_run(c, tid, run["run_id"])
    check(
        run["status"] == "interrupted", f"run status = {run['status']} (expected interrupted before erp-journal-post)"
    )

    save_state(hitl_thread_id=tid, hitl_run_id=run["run_id"])
    state = await c.threads.get_state(tid)
    # Aegra reports interrupts both at the top level and per task; accept either.
    interrupts = list(state.get("interrupts") or []) or [
        i for t in state.get("tasks", []) for i in t.get("interrupts", [])
    ]
    check(len(interrupts) >= 1, f"thread has {len(interrupts)} pending interrupt(s); next={state.get('next')}")
    payload = interrupts[0].get("value") if interrupts else None
    print("interrupt payload:", str(payload)[:300])
    writes = [e for e in events_from_log() if e["event_type"] == "tool.call" and e["tool"]["name"] == WRITE]
    check(not writes, f"no {WRITE} tool.call event before approval")


async def resume() -> None:
    """Approve and finish (works after a server restart: state lives in the checkpointer)."""
    c = client()
    st = load_state()
    tid = st["hitl_thread_id"]
    before = len([e for e in events_from_log() if e["event_type"] == "tool.call" and e["tool"]["name"] == WRITE])
    run = await c.runs.create(
        tid,
        assistant_id_for(AGENT_ID),
        command={"resume": {"decisions": [{"type": "approve"}]}},
        config={"configurable": run_configurable(AGENT_ID)},
    )
    check(run["run_id"] != st["hitl_run_id"], "resume is a NEW run on the same thread")
    run = await wait_run(c, tid, run["run_id"])
    check(run["status"] == "success", f"resumed run status = {run['status']}")

    state = await c.threads.get_state(tid)
    text = last_ai_text(state["values"]["messages"])
    check("JE-1000-2026-10-" in text, f"final answer after approval: {text[:100]!r}")
    after = [e for e in events_from_log() if e["event_type"] == "tool.call" and e["tool"]["name"] == WRITE]
    check(len(after) == before + 1 and after[-1]["tool"]["allowed"], f"exactly one allowed {WRITE} tool.call recorded")
    check(
        state["values"]["asset_versions"].get(f"tool://{WRITE}@active", "").endswith("@1"),
        "pinned tool version kept",
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    asyncio.run(start() if mode == "start" else resume())
