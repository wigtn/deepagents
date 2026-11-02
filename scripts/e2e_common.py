"""Shared helpers for the e2e scripts (Aegra must be running on AEGRA_URL, default localhost:2026).

Every run request sends the FULL ``configurable`` (agent_id included): Aegra merges the
assistant's config and the run's config shallowly, so a partial configurable would drop
``agent_id`` (see docs/notes/00-field-notes.md, aegra #2).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AEGRA_URL = os.environ.get("AEGRA_URL", "http://localhost:2026")
GRAPH_ID = "worker"
ASSISTANT_NS = uuid.UUID("7d1a2c3e-5f60-4b7a-9c8d-0e1f2a3b4c5d")  # never change: assistant ids derive from it
STATE_FILE = ROOT / "var" / "e2e_state.json"


def assistant_id_for(agent_id: str) -> str:
    """Deterministic assistant id per manifest (idempotent registration)."""
    return str(uuid.uuid5(ASSISTANT_NS, agent_id))


def run_configurable(agent_id: str, **extra: Any) -> dict[str, Any]:
    """Complete configurable for a run. Always includes agent_id."""
    base: dict[str, Any] = {
        "agent_id": agent_id,
        "session_type": "production",
        "entity": "1000",
        "data_scope": ["erp:read"],
    }
    base.update(extra)
    return base


def client() -> LangGraphClient:
    """Async SDK client against the local runtime."""
    return get_client(url=AEGRA_URL)


def save_state(**values: Any) -> None:
    """Persist ids between script invocations (e.g. before/after a server restart)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = load_state()
    current.update(values)
    STATE_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")


def load_state() -> dict[str, Any]:
    """Read persisted ids."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def last_ai_text(messages: list[dict[str, Any]]) -> str:
    """Text of the last assistant message in a thread state."""
    for m in reversed(messages):
        if m.get("type") == "ai":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


async def wait_run(c: LangGraphClient, thread_id: str, run_id: str, timeout: float = 120.0) -> dict[str, Any]:
    """Poll the run record until it leaves pending/running."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        run = await c.runs.get(thread_id, run_id)
        if run["status"] not in ("pending", "running"):
            return run
        if loop.time() > deadline:
            raise TimeoutError(f"run {run_id} still {run['status']}")
        await asyncio.sleep(0.5)


def events_from_log(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Read the worker's JSONL event log (written by EventRecorderMiddleware)."""
    log = path or Path(os.environ.get("WIGTN_AGENT_EVENT_LOG") or ROOT / "var" / "events.jsonl")
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(condition: bool, message: str) -> None:
    """Assert with a readable failure (scripts exit non-zero)."""
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"ok  : {message}")
