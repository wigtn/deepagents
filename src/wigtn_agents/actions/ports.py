"""Gather and writeback ports of the action lane. Both are code, selected by name.

* A **gatherer** collects the inputs the model needs before it reasons. The model has no say in
  what is fetched; a gatherer that cannot get its inputs fails the run.
* A **writeback** port stores the validated result. Writing is never an LLM tool in this lane.

Built-ins: gatherers ``input`` (the run input as is) and ``default-plan`` (calls the skill's
``default_plan`` read tools, see ``chassis``); writebacks ``none`` and ``jsonl``. Packages add more
with entry points:

```toml
[project.entry-points."wigtn_agents.gatherers"]
my-source = "my_package.ports:gather"        # gather(request: GatherRequest) -> dict
[project.entry-points."wigtn_agents.writebacks"]
my-store = "my_package.ports:write"          # write(request: WritebackRequest) -> dict
```

A port may be a plain function or a coroutine function.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Optional

from wigtn_agents.context import RunContext
from wigtn_agents.manifest import ActionManifest

GATHERER_GROUP = "wigtn_agents.gatherers"
WRITEBACK_GROUP = "wigtn_agents.writebacks"
ACTION_LOG_ENV = "WIGTN_AGENT_ACTION_LOG"


@dataclass(frozen=True)
class GatherRequest:
    """What a gatherer gets."""

    action: ActionManifest
    input: dict[str, Any]
    ctx: RunContext
    tools: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WritebackRequest:
    """What a writeback port gets: the result already validated against ``output_schema``."""

    action: ActionManifest
    result: dict[str, Any]
    ctx: RunContext
    run_id: Optional[str]


Gatherer = Callable[[GatherRequest], Any]
Writeback = Callable[[WritebackRequest], Any]


def gather_input(request: GatherRequest) -> dict[str, Any]:
    """The run input, unchanged."""
    return dict(request.input)


def write_none(request: WritebackRequest) -> dict[str, Any]:
    """Store nothing; the result stays in thread state."""
    return {"state": "skipped"}


def write_jsonl(request: WritebackRequest) -> dict[str, Any]:
    """Append the result to ``WIGTN_AGENT_ACTION_LOG`` (``var/actions.jsonl``)."""
    path = Path(os.environ.get(ACTION_LOG_ENV, "var/actions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "action": request.action.agent_id,
        "action_version": request.action.metadata.version,
        "run_id": request.run_id,
        "batch_id": request.ctx.get("batch_id"),
        "result": request.result,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return {"state": "written", "path": str(path)}


_local_gatherers: dict[str, Gatherer] = {"input": gather_input}
_local_writebacks: dict[str, Writeback] = {"none": write_none, "jsonl": write_jsonl}


def register_gatherer(name: str, gatherer: Gatherer) -> None:
    """Register in-process (tests, notebooks)."""
    _local_gatherers[name] = gatherer


def register_writeback(name: str, writeback: Writeback) -> None:
    """Register in-process (tests, notebooks)."""
    _local_writebacks[name] = writeback


def _lookup(name: str, group: str, local: dict[str, Any]) -> Any:
    if name in local:
        return local[name]
    for ep in entry_points(group=group):
        if ep.name == name:
            return ep.load()
    raise KeyError(f"등록되지 않은 port입니다: {name} (entry point 그룹 {group})")


def gatherer_for(name: str) -> Gatherer:
    """Gatherer by name; ``default-plan`` is resolved by the chassis."""
    return _lookup(name, GATHERER_GROUP, _local_gatherers)  # type: ignore[no-any-return]


def writeback_for(name: str) -> Writeback:
    """Writeback port by name."""
    return _lookup(name, WRITEBACK_GROUP, _local_writebacks)  # type: ignore[no-any-return]


async def call_port(port: Callable[[Any], Any], request: Any) -> dict[str, Any]:
    """Call a sync or async port and insist on a dict."""
    result = port(request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError(f"port는 dict를 돌려줘야 합니다: {type(result).__name__}")
    return result
