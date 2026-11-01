"""Catalog → Aegra assistants, one-way derived registration (D-B1).

The catalog stays the source of truth; each agent manifest becomes an assistant on graph
``worker`` and each action manifest one on graph ``action``. The config carries only
``agent_id`` (asset bodies are loaded per run). Re-running is idempotent: ids are uuid5 of the
agent_id, existing assistants are updated in place.
"""

from __future__ import annotations

import asyncio

from e2e_common import GRAPH_ID, assistant_id_for, client
from wigtn_agents.worker import load_catalog

ACTION_GRAPH_ID = "action"


async def main() -> None:
    """Register or update every agent and action manifest."""
    c = client()
    catalog = load_catalog()
    entries = [(GRAPH_ID, a, catalog.manifest(a)) for a in catalog.agent_ids()]
    entries += [(ACTION_GRAPH_ID, a, catalog.action(a)) for a in catalog.action_ids()]
    existing: set[str] = set()
    for graph_id in (GRAPH_ID, ACTION_GRAPH_ID):
        existing |= {a["assistant_id"] for a in await c.assistants.search(graph_id=graph_id, limit=100)}
    for graph_id, agent_id, manifest in entries:
        aid = assistant_id_for(agent_id)
        config = {"configurable": {"agent_id": agent_id}}
        metadata = {
            "agent_id": agent_id,
            "kind": manifest.kind,
            "manifest_version": manifest.metadata.version,
            "source": "catalog",
        }
        if aid in existing:
            await c.assistants.update(aid, graph_id=graph_id, config=config, metadata=metadata, name=agent_id)
            print(f"updated  {graph_id:6s} {agent_id:18s} {aid}")
        else:
            await c.assistants.create(
                graph_id=graph_id,
                config=config,
                metadata=metadata,
                assistant_id=aid,
                name=agent_id,
                if_exists="do_nothing",
            )
            print(f"created  {graph_id:6s} {agent_id:18s} {aid}")


if __name__ == "__main__":
    asyncio.run(main())
