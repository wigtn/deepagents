"""e2e (a): one records-guide run through Aegra — stream, final answer, pinned versions in thread state."""

from __future__ import annotations

import asyncio

from e2e_common import assistant_id_for, check, client, last_ai_text, run_configurable, wait_run


async def main() -> None:
    """Create thread → run → stream → inspect state."""
    c = client()
    agent_id = "records-guide"
    thread = await c.threads.create()
    tid = thread["thread_id"]
    seen: list[str] = []
    async for chunk in c.runs.stream(
        tid,
        assistant_id_for(agent_id),
        input={"messages": [{"role": "user", "content": "8월 기록 점검 부탁해 (범위 1000)"}]},
        config={"configurable": run_configurable(agent_id)},
        stream_mode=["updates"],
    ):
        seen.append(chunk.event)
    check(len(seen) > 0, f"streamed {len(seen)} events ({sorted(set(seen))})")

    runs = await c.runs.list(tid)
    run = await wait_run(c, tid, runs[0]["run_id"])
    check(run["status"] == "success", f"run status = {run['status']}")

    state = await c.threads.get_state(tid)
    values = state["values"]
    check("asset_versions" in values, "thread state carries asset_versions (pinned at run start)")
    check(values["asset_versions"].get("skill://example/checklist@active", "").endswith("@3"), "skill pinned @3")
    text = last_ai_text(values["messages"])
    check(bool(text), f"final answer: {text[:80]!r}")


if __name__ == "__main__":
    asyncio.run(main())
