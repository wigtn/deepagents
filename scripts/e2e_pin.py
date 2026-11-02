"""e2e (d): activating skill v2 mid-thread does not change a thread pinned at v1.

Edits the catalog files in place (v1 → retired, adds v2 active) and restores them at the end.
The worker re-reads the catalog when file mtimes change, without a restart.
"""

from __future__ import annotations

import asyncio
import shutil

import yaml

from e2e_common import ROOT, assistant_id_for, check, client, run_configurable, wait_run

AGENT_ID = "erp-assistant"
SKILL = "skill://erp/lookup@active"
V3 = ROOT / "catalog" / "assets" / "skill.erp.lookup.yaml"  # pinned version (v1)
V4 = ROOT / "catalog" / "assets" / "skill.erp.lookup.v2.yaml"  # the version activated mid-thread
BACKUP = ROOT / "var" / "skill.erp.lookup.yaml.bak"


def activate_v4() -> None:
    """v1 retired, v2 active (immutable versions: v2 is a new file)."""
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(V3, BACKUP)
    data = yaml.safe_load(V3.read_text(encoding="utf-8"))
    v4 = dict(data)
    v4["version"] = 2
    v4["description"] = data.get("description", "") + " (v2)"
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
    """Pin on thread A at v1, activate v2, continue A (still v1), start B (v2)."""
    c = client()
    a = (await c.threads.create())["thread_id"]
    values = await one_turn(a, "PO-4500000103 입고 얼마나 됐어?")
    check(values["asset_versions"][SKILL].endswith("@1"), "thread A pinned @1")
    try:
        activate_v4()
        values = await one_turn(a, "이어서 진행해줘")
        check(
            values["asset_versions"][SKILL].endswith("@1"),
            "thread A still @1 after v2 went active",
        )
        b = (await c.threads.create())["thread_id"]
        values_b = await one_turn(b, "새 스레드에서 조회")
        check(values_b["asset_versions"][SKILL].endswith("@2"), "new thread B pins @2")
    finally:
        restore()
        print("catalog restored")


if __name__ == "__main__":
    asyncio.run(main())
