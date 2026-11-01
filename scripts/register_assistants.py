"""Catalog → Aegra assistants, one-way derived registration (D-B1).

The catalog stays the source of truth; each manifest becomes an assistant on graph ``worker``
whose config carries only ``agent_id`` (asset bodies are loaded per run, D-B2). Re-running is
idempotent: ids are uuid5 of the agent_id, existing assistants are updated in place.
"""

from __future__ import annotations

import asyncio

from e2e_common import GRAPH_ID, assistant_id_for, client
from wigtn_agents.worker import load_catalog


async def main() -> None:
    """Register or update every manifest."""
    c = client()
    catalog = load_catalog()
    existing = {a["assistant_id"] for a in await c.assistants.search(graph_id=GRAPH_ID, limit=100)}
    for agent_id, manifest in sorted(catalog._manifests.items()):
        aid = assistant_id_for(agent_id)
        config = {"configurable": {"agent_id": agent_id}}
        metadata = {
            "agent_id": agent_id,
            "manifest_version": manifest.metadata.version,
            "worker_class": manifest.metadata.worker_class.value,
            "source": "catalog",
        }
        if aid in existing:
            await c.assistants.update(aid, graph_id=GRAPH_ID, config=config, metadata=metadata, name=agent_id)
            print(f"updated  {agent_id:18s} {aid}")
        else:
            await c.assistants.create(
                graph_id=GRAPH_ID,
                config=config,
                metadata=metadata,
                assistant_id=aid,
                name=agent_id,
                if_exists="do_nothing",
            )
            print(f"created  {agent_id:18s} {aid}")


if __name__ == "__main__":
    asyncio.run(main())
