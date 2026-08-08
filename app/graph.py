"""Memory graph (console "Graph" view): derived entirely from existing
relational columns — Fact.source_event_ids and Fact.supersedes_id — no
separate graph store. A fact IS a node the moment it exists; an edge is
just those foreign keys read back out.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Event, Fact, FactStatus
from app.schemas.graph import GraphEdge, GraphNode, GraphResponse


def _fact_label(fact: Fact) -> str:
    value = fact.value.get("text") if isinstance(fact.value, dict) else None
    return f"{fact.predicate}: {value}" if value else fact.predicate


async def build_subject_graph(
    session: AsyncSession, *, project_id: str, subject_id: str
) -> GraphResponse:
    facts = list(
        (
            await session.execute(
                select(Fact)
                .where(
                    Fact.project_id == project_id,
                    Fact.subject_id == subject_id,
                    Fact.status != FactStatus.deleted,
                )
                .order_by(Fact.recorded_from)
            )
        )
        .scalars()
        .all()
    )

    event_ids: set[uuid.UUID] = set()
    for fact in facts:
        event_ids.update(fact.source_event_ids)
    events = (
        list(
            (
                await session.execute(
                    select(Event).where(Event.id.in_(event_ids))
                )
            )
            .scalars()
            .all()
        )
        if event_ids
        else []
    )

    subject_node_id = f"subject:{subject_id}"
    nodes = [
        GraphNode(id=subject_node_id, kind="subject", label=subject_id, meta={"facts": len(facts)})
    ]
    edges: list[GraphEdge] = []

    fact_node_id = {fact.id: f"fact:{fact.id}" for fact in facts}
    for fact in facts:
        node_id = fact_node_id[fact.id]
        nodes.append(
            GraphNode(
                id=node_id,
                kind="fact",
                label=_fact_label(fact),
                status=fact.status.value,
                meta={
                    "confidence": fact.confidence,
                    "version": fact.version,
                    "recorded_from": fact.recorded_from.isoformat(),
                },
            )
        )
        edges.append(GraphEdge(source=subject_node_id, target=node_id, kind="has_fact"))
        if fact.supersedes_id and fact.supersedes_id in fact_node_id:
            edges.append(
                GraphEdge(
                    source=node_id,
                    target=fact_node_id[fact.supersedes_id],
                    kind="supersedes",
                )
            )

    event_node_id = {event.id: f"event:{event.id}" for event in events}
    for event in events:
        nodes.append(
            GraphNode(
                id=event_node_id[event.id],
                kind="event",
                label=event.kind,
                meta={"occurred_at": event.occurred_at.isoformat()},
            )
        )
    for fact in facts:
        for eid in fact.source_event_ids:
            if eid in event_node_id:
                edges.append(
                    GraphEdge(
                        source=fact_node_id[fact.id],
                        target=event_node_id[eid],
                        kind="derived_from",
                    )
                )

    return GraphResponse(
        subject_id=subject_id, project_id=project_id, nodes=nodes, edges=edges
    )
