"""Extra middleware: the slot where research systems attach their own code.

A research package registers a factory under the entry point group ``wigtn_agents.middleware``:

```toml
[project.entry-points."wigtn_agents.middleware"]
my-defense = "my_package.middleware:build"   # build(spec: MiddlewareSpec) -> AgentMiddleware
```

A manifest lists names in ``extra_middleware``; a run adds names with ``configurable.middleware``.
Only registered names can be used: the image decides what code runs, a request only selects it.
The slot is right inside the event recorder, so the record shows what the extra middleware did
(a denied call, a rewritten argument), and outside the permission gate.

Prefer ``wrap_model_call`` / ``wrap_tool_call`` hooks. ``before_*`` / ``after_*`` hooks add graph
nodes; the schema-only reader includes every registered factory so threads stay readable, which
means a factory must also build without a real run (``spec.schema_only`` is true then).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from wigtn_agents.context import RunContext
from wigtn_agents.manifest import AnyManifest

ENTRY_POINT_GROUP = "wigtn_agents.middleware"


@dataclass(frozen=True)
class MiddlewareSpec:
    """What a factory gets: the manifest and run context it is attached to."""

    manifest: AnyManifest
    ctx: RunContext
    schema_only: bool = False


MiddlewareFactory = Callable[[MiddlewareSpec], AgentMiddleware]
_local: dict[str, MiddlewareFactory] = {}


def register_middleware(name: str, factory: MiddlewareFactory) -> None:
    """Register in-process (tests, notebooks). Entry points are the packaged path."""
    _local[name] = factory


def unregister_middleware(name: str) -> None:
    """Remove an in-process registration."""
    _local.pop(name, None)


def _as_factory(obj: Any) -> MiddlewareFactory:
    if isinstance(obj, type) and issubclass(obj, AgentMiddleware):
        cls = obj
        return lambda spec: cls()
    if callable(obj):
        return obj  # type: ignore[no-any-return]
    raise TypeError(f"미들웨어 factory가 아닙니다: {obj!r}")


def available_middleware() -> dict[str, MiddlewareFactory]:
    """Entry-point factories plus in-process ones (in-process wins on a name clash)."""
    found: dict[str, MiddlewareFactory] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        found[ep.name] = _as_factory(ep.load())
    found.update(_local)
    return found


def merge_names(*groups: Iterable[str]) -> tuple[str, ...]:
    """Names in first-seen order, without duplicates."""
    seen: dict[str, None] = {}
    for group in groups:
        for name in group:
            seen.setdefault(name, None)
    return tuple(seen)


def build_extra_middleware(names: Iterable[str], spec: MiddlewareSpec) -> list[AgentMiddleware]:
    """Instantiate the named middleware. Unknown names fail; nothing is loaded from a request."""
    names = tuple(names)
    if not names:
        return []
    factories = available_middleware()
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise KeyError(f"등록되지 않은 미들웨어입니다: {unknown} (entry point 그룹 {ENTRY_POINT_GROUP})")
    return [factories[n](spec) for n in names]
