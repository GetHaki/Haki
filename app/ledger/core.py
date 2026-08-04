"""Memory Ledger: durable capture of events and versioned memory objects.

Small interface on purpose: write an event, read an object, list a timeline,
apply a versioned mutation (PRD — "Modules profonds").
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ApiError
from app.models import Event, Fact, FactStatus
from app.schemas import EventIn


def compute_event_hash(event: EventIn) -> str:
    """sha256 of the canonical business content of an event."""
    canonical = json.dumps(
        {
            "org_id": event.org_id,
            "project_id": event.project_id,
            "subject_type": event.subject_type,
            "subject_id": event.subject_id,
            "kind": event.kind,
            "occurred_at": event.occurred_at.isoformat(),
            "payload": event.payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _validate_scope(event: EventIn, index: int) -> None:
    if not event.subject_id:
        raise ApiError(
            type="missing_scope",
            message="subject_id is required on every event",
            field=f"events.{index}.subject_id",
        )


async def write_events(
    session: AsyncSession,
    events: list[EventIn],
    batch_idempotency_key: str | None = None,
) -> list[tuple[Event, bool]]:
    """Idempotent insert: (project_id, idempotency_key) is unique.

    Returns (event, deduplicated) pairs in request order. An event whose key
    already exists is not re-inserted; the existing row is returned instead.
    """
    for i, event in enumerate(events):
        _validate_scope(event, i)

    rows = []
    for event in events:
        content_hash = compute_event_hash(event)
        # A batch-level key is namespaced per event content so several events
        # in one batch never collide on (project_id, idempotency_key).
        if batch_idempotency_key:
            key = f"{batch_idempotency_key}:{content_hash}"
        elif event.idempotency_key:
            key = event.idempotency_key
        else:
            key = f"{content_hash}:{event.subject_id}"
        rows.append(
            {
                "org_id": event.org_id,
                "project_id": event.project_id,
                "subject_type": event.subject_type,
                "subject_id": event.subject_id,
                "actor_type": event.actor_type,
                "actor_id": event.actor_id,
                "agent_id": event.agent_id,
                "thread_id": event.thread_id,
                "run_id": event.run_id,
                "kind": event.kind,
                "occurred_at": event.occurred_at,
                "payload": event.payload,
                "source": event.source,
                "classification": event.classification,
                "retention_policy": event.retention_policy,
                "hash": content_hash,
                "idempotency_key": key,
            }
        )

    stmt = (
        pg_insert(Event)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_events_idempotency")
        .returning(Event.id, Event.idempotency_key)
    )
    inserted = {
        key: id_ for id_, key in (await session.execute(stmt)).all()  # noqa: A002
    }

    # Re-select so every caller gets the full rows, inserted or pre-existing.
    keys = [row["idempotency_key"] for row in rows]
    existing = (
        (
            await session.execute(
                select(Event).where(Event.idempotency_key.in_(keys))
            )
        )
        .scalars()
        .all()
    )
    by_key = {event.idempotency_key: event for event in existing}

    return [(by_key[row["idempotency_key"]], row["idempotency_key"] not in inserted) for row in rows]


async def list_timeline(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
) -> list[Event]:
    """Events of one subject within one project, business-time ordered.

    Scope filtering is mandatory: callers must always pass both ids.
    """
    stmt = (
        select(Event)
        .where(Event.project_id == project_id, Event.subject_id == subject_id)
        .order_by(Event.occurred_at, Event.recorded_at)
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_fact(session: AsyncSession, fact_id: uuid.UUID) -> Fact:
    fact = await session.get(Fact, fact_id)
    if fact is None:
        raise ApiError(
            type="fact_not_found",
            message=f"Fact {fact_id} does not exist",
            field="fact_id",
            status_code=404,
        )
    return fact


async def create_fact(
    session: AsyncSession,
    *,
    org_id: str,
    project_id: str,
    subject_id: str,
    predicate: str,
    value: dict,
    subject_type: str = "user",
    agent_id: str | None = None,
    qualifiers: dict | None = None,
    confidence: float | None = None,
    valid_from: datetime | None = None,
    source_event_ids: list[uuid.UUID] | None = None,
    supersedes_id: uuid.UUID | None = None,
) -> Fact:
    fact = Fact(
        org_id=org_id,
        project_id=project_id,
        subject_type=subject_type,
        subject_id=subject_id,
        agent_id=agent_id,
        predicate=predicate,
        value=value,
        qualifiers=qualifiers or {},
        status=FactStatus.candidate,
        confidence=confidence,
        valid_from=valid_from,
        source_event_ids=source_event_ids or [],
        supersedes_id=supersedes_id,
        version=1,
    )
    session.add(fact)
    await session.flush()
    return fact


# Explicit status lifecycle (PRD — "Modèle de mémoire et règles de cycle de vie").
# candidate -> superseded exists for conflict resolution (sprint 6): the
# losing fact of a conflict set is typically still a candidate, and
# resolving the set supersedes it (with supersedes_id on the kept fact).
ALLOWED_TRANSITIONS: dict[FactStatus, set[FactStatus]] = {
    FactStatus.candidate: {
        FactStatus.active,
        FactStatus.superseded,
        FactStatus.disputed,
        FactStatus.disabled,
        FactStatus.deleted,
    },
    FactStatus.active: {
        FactStatus.superseded,
        FactStatus.disputed,
        FactStatus.disabled,
        FactStatus.deleted,
    },
    FactStatus.superseded: {FactStatus.disputed, FactStatus.deleted},
    FactStatus.disputed: {
        FactStatus.active,
        FactStatus.superseded,
        FactStatus.disabled,
        FactStatus.deleted,
    },
    FactStatus.disabled: {FactStatus.active, FactStatus.deleted},
    FactStatus.deleted: set(),  # terminal: deleted -> * is forbidden
}


class IllegalTransitionError(ApiError):
    def __init__(self, current: FactStatus, target: FactStatus) -> None:
        super().__init__(
            type="illegal_status_transition",
            message=f"Fact status cannot transition from {current.value} to {target.value}",
            field="status",
        )


async def transition_fact_status(
    session: AsyncSession,
    fact_id: uuid.UUID,
    target: FactStatus,
) -> Fact:
    """Apply a status mutation, enforcing the lifecycle graph above."""
    fact = await get_fact(session, fact_id)
    if target not in ALLOWED_TRANSITIONS[fact.status]:
        raise IllegalTransitionError(fact.status, target)

    fact.status = target
    fact.version += 1
    if target is FactStatus.deleted:
        fact.recorded_to = datetime.now(timezone.utc)
    await session.flush()
    return fact
