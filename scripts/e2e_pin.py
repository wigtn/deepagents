"""e2e (d): activating skill v4 mid-thread does not change a thread pinned at v3.

Edits the catalog files in place (v3 → retired, adds v4 active) and restores them at the end.
The worker re-reads the catalog when file mtimes change, without a restart.
"""

from __future__ import annotations

import asyncio
import shutil

import yaml

from e2e_common import ROOT, assistant_id_for, check, client, run_configurable, wait_run

AGENT_ID = "records-guide"
V3 = ROOT / "catalog" / "assets" / "skill.example.checklist.yaml"
V4 = ROOT / "catalog" / "assets" / "skill.example.checklist.v4.yaml"
BACKUP = ROOT / "var" / "skill.example.checklist.yaml.bak"


def activate_v4() -> None:
    """v3 retired, v4 active (immutable versions: v4 is a new file)."""
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(V3, BACKUP)
    data = yaml.safe_load(V3.read_text(encoding="utf-8"))
    v4 = dict(data)
    v4["version"] = 4
    v4["description"] = data.get("description", "") + " (v4)"
    V4.write_text(yaml.safe_dump(v4, allow_unicode=True, sort_keys=False), encoding="utf-8")
    data["status"] = "retired"
    V3.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def restore() -> None:
    """Put the catalog back."""
    if BACKUP.exists():
        shutil.move(BACKUP, V3)
    if V4.exists():
        V4.unlink()


async def one_turn(thread_id: str, text: str) -> dict:
    c = client()
    run = await c.runs.create(
        thread_id,
        assistant_id_for(AGENT_ID),
        input={"messages": [{"role": "user", "content": text}]},
        config={"configurable": run_configurable(AGENT_ID)},
    )
    run = await wait_run(c, thread_id, run["run_id"])
    check(run["status"] == "success", f"turn on {thread_id[:8]}: {run['status']}")
    return (await c.threads.get_state(thread_id))["values"]


async def main() -> None:
    """Pin on thread A at v3, activate v4, continue A (still v3), start B (v4)."""
    c = client()
    a = (await c.threads.create())["thread_id"]
    values = await one_turn(a, "8월 기록 점검 (범위 1000)")
    check(values["asset_versions"]["skill://example/checklist@active"].endswith("@3"), "thread A pinned @3")
    try:
        activate_v4()
        values = await one_turn(a, "이어서 진행해줘")
        check(
            values["asset_versions"]["skill://example/checklist@active"].endswith("@3"),
            "thread A still @3 after v4 went active",
        )
        b = (await c.threads.create())["thread_id"]
        values_b = await one_turn(b, "새 스레드에서 점검")
        check(values_b["asset_versions"]["skill://example/checklist@active"].endswith("@4"), "new thread B pins @4")
    finally:
        restore()
        print("catalog restored")


if __name__ == "__main__":
    asyncio.run(main())
