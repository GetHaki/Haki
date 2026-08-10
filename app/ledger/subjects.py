"""Identity resolution (M4): channel aliases -> canonical subject, and
subject merge with a timestamped receipt.

Design rules (PRD coherence):
- Resolution is a mapping requested by the CLIENT backend (HTTP field or
  dedicated endpoint) — the model never chooses a scope, so the MCP tools
  expose none of this.
- Registration is race-safe: INSERT .. ON CONFLICT DO NOTHING + re-select
  (same pattern as ledger.write_events), so two concurrent captures of the
  same brand-new alias converge on one row.
- A merge re-scopes events, facts, conflict_sets and context_traces in the
  caller's transaction, journals a SubjectMergeReceipt (counters + exact
  moved ids), re-points every alias of the source and leaves a
  "subject"-kind tombstone so the old id stays resolvable and the
  fragmentation detector can warn about it.
"""

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ApiError
from app.models import ConflictSet, ContextTrace, Event, Fact, SubjectAlias, SubjectMergeReceipt

# Reserved kind for merge tombstones: after merging src -> tgt, the row
# (kind="subject", value=src) -> tgt keeps the old id resolvable.
TOMBSTONE_KIND = "subject"


async def resolve_alias(
    session: AsyncSession,
    *,
    project_id: str,
    alias_kind: str,
    alias_value: str,
    canonical_subject_id: str | None = None,
    register: bool = True,
) -> tuple[SubjectAlias | None, bool]:
    """Resolve (and optionally register) one alias. Returns (row, created).

    - Existing row + a DIFFERENT explicit canonical -> 409 alias_conflict:
      re-pointing an alias is merge's job, never a silent side effect.
    - Missing row, register=True: canonical defaults to the deterministic
      "{kind}:{value}" (self-registration; the kind prefix prevents raw-value
      collisions between channels). If that derived id exceeds 128 chars the
      caller must provide canonical_subject_id (422
      alias_self_registration_too_long) — subject_id columns are 128 wide.
    - Missing row, register=False: returns (None, False) — read paths never
      write.
    """
    stmt = select(SubjectAlias).where(
        SubjectAlias.project_id == project_id,
        SubjectAlias.alias_kind == alias_kind,
        SubjectAlias.alias_value == alias_value,
    )
    row = (await session.execute(stmt)).scalars().first()

    if row is not None:
        if (
            canonical_subject_id is not None
            and canonical_subject_id != row.canonical_subject_id
        ):
            raise ApiError(
                type="alias_conflict",
                message=(
                    "alias already resolves to a different subject; use "
                    "POST /v1/subjects/merge to unify subjects"
                ),
                field="canonical_subject_id",
                status_code=409,
            )
        return row, False

    if not register:
        return None, False

    canonical = canonical_subject_id or f"{alias_kind}:{alias_value}"
    if canonical_subject_id is None and len(canonical) > 128:
        raise ApiError(
            type="alias_self_registration_too_long",
            message=(
                "derived canonical id exceeds 128 characters; provide "
                "canonical_subject_id explicitly"
            ),
            field="canonical_subject_id",
        )

    insert_stmt = (
        pg_insert(SubjectAlias)
        .values(
            project_id=project_id,
            alias_kind=alias_kind,
            alias_value=alias_value,
            canonical_subject_id=canonical,
        )
        .on_conflict_do_nothing(constraint="uq_subject_aliases_identity")
        .returning(SubjectAlias.id)
    )
    inserted_id = (await session.execute(insert_stmt)).scalar_one_or_none()

    row = (await session.execute(stmt)).scalars().first()
    created = inserted_id is not None and row is not None and row.id == inserted_id

    if row is not None and canonical_subject_id is not None and row.canonical_subject_id != canonical:
        raise ApiError(
            type="alias_conflict",
            message=(
                "alias already resolves to a different subject; use "
                "POST /v1/subjects/merge to unify subjects"
            ),
            field="canonical_subject_id",
            status_code=409,
        )

    return row, created


async def merge_subjects(
    session: AsyncSession,
    *,
    project_id: str,
    source_subject_id: str,
    target_subject_id: str,
) -> tuple[SubjectMergeReceipt, dict[str, Any]]:
    """Merge source into target within one project, in the caller's
    transaction. Returns (receipt, counters).

    Re-scopes: events, facts, conflict_sets, context_traces (traces
    included on purpose: after a merge, forgetting the TARGET must erase
    everything the merged identity ever produced — leaving traces under the
    dead source id would make erasure incomplete). Does NOT touch
    forget_receipts (a receipt is a journal entry, never rewritten), jobs
    or feedback (no subject_id column). Does NOT adjudicate contradictions:
    facts from both subjects coexist under target until normal supersession
    /conflict machinery deals with them — merging is re-scoping, not
    semantic reconciliation.

    Typed error: invalid_merge when source == target or either is empty.
    """
    if (
        not source_subject_id
        or not target_subject_id
        or source_subject_id == target_subject_id
    ):
        raise ApiError(
            type="invalid_merge",
            message=(
                "source_subject_id and target_subject_id must be two "
                "different non-empty subjects"
            ),
            field="target_subject_id",
        )

    moved_events = [
        str(row_id)
        for (row_id,) in (
            await session.execute(
                update(Event)
                .where(Event.project_id == project_id, Event.subject_id == source_subject_id)
                .values(subject_id=target_subject_id)
                .returning(Event.id)
            )
        ).all()
    ]
    moved_facts = [
        str(row_id)
        for (row_id,) in (
            await session.execute(
                update(Fact)
                .where(Fact.project_id == project_id, Fact.subject_id == source_subject_id)
                .values(subject_id=target_subject_id)
                .returning(Fact.id)
            )
        ).all()
    ]
    moved_conflicts = [
        str(row_id)
        for (row_id,) in (
            await session.execute(
                update(ConflictSet)
                .where(
                    ConflictSet.project_id == project_id,
                    ConflictSet.subject_id == source_subject_id,
                )
                .values(subject_id=target_subject_id)
                .returning(ConflictSet.id)
            )
        ).all()
    ]
    moved_traces = [
        str(row_id)
        for (row_id,) in (
            await session.execute(
                update(ContextTrace)
                .where(
                    ContextTrace.project_id == project_id,
                    ContextTrace.subject_id == source_subject_id,
                )
                .values(subject_id=target_subject_id)
                .returning(ContextTrace.id)
            )
        ).all()
    ]

    repointed = [
        row_id
        for (row_id,) in (
            await session.execute(
                update(SubjectAlias)
                .where(
                    SubjectAlias.project_id == project_id,
                    SubjectAlias.canonical_subject_id == source_subject_id,
                )
                .values(canonical_subject_id=target_subject_id)
                .returning(SubjectAlias.id)
            )
        ).all()
    ]

    await session.execute(
        pg_insert(SubjectAlias)
        .values(
            project_id=project_id,
            alias_kind=TOMBSTONE_KIND,
            alias_value=source_subject_id,
            canonical_subject_id=target_subject_id,
        )
        .on_conflict_do_nothing(constraint="uq_subject_aliases_identity")
    )

    counters = {
        "events_moved": len(moved_events),
        "facts_moved": len(moved_facts),
        "conflict_sets_moved": len(moved_conflicts),
        "traces_moved": len(moved_traces),
        "aliases_repointed": len(repointed),
    }
    receipt = SubjectMergeReceipt(
        project_id=project_id,
        source_subject_id=source_subject_id,
        target_subject_id=target_subject_id,
        counters=counters,
        moved={
            "events": moved_events,
            "facts": moved_facts,
            "conflict_sets": moved_conflicts,
            "traces": moved_traces,
            "aliases_repointed": [str(i) for i in repointed],
        },
    )
    session.add(receipt)
    await session.flush()
    return receipt, counters
