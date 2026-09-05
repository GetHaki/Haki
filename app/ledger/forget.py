"""Forget operations (sprint 4): disable or erase memory, with a receipt.

Two scopes, exactly one per call:

- `fact_id`   -> one fact transitions to `disabled` or `deleted` through the
  Ledger lifecycle (deleted also sets `recorded_to`: bitemporal erasure);
- `subject_id` -> the whole subject within the project:
  * `disable`: every active/candidate fact transitions to `disabled`
    (reversible, history kept);
  * `delete`: REAL deletion of the subject's facts (embeddings go with
    them), conflict sets, events (episode chunks follow via ON DELETE
    CASCADE), context traces, learned predicate aliases, subject aliases
    resolving to it, and feedback attached to its facts (free-text comments
    may carry personal data -- the observation must not outlive the erasure).

Every operation is journaled in `forget_receipts` (embryo of the PRD
erasure receipt) and the counters say what actually happened.
"""

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ApiError
from app.ledger.core import get_fact, transition_fact_status
from app.models import (
    ConflictSet,
    ContextTrace,
    Event,
    Fact,
    FactStatus,
    Feedback,
    ForgetReceipt,
    PredicateAlias,
    SubjectAlias,
)

VALID_MODES = ("disable", "delete")


async def forget(
    session: AsyncSession,
    *,
    project_id: str,
    mode: str,
    fact_id: uuid.UUID | None = None,
    subject_id: str | None = None,
) -> tuple[ForgetReceipt, dict[str, Any]]:
    """Apply a forget operation. Returns (receipt, counters).

    Typed errors: `invalid_forget_scope` (exactly one of fact_id/subject_id,
    or an unknown mode) and `fact_not_found` (unknown fact_id, or a fact
    outside the given project — same error, no cross-scope leak).
    """
    if mode not in VALID_MODES:
        raise ApiError(
            type="invalid_forget_scope",
            message=f"mode must be one of {VALID_MODES}",
            field="mode",
        )
    if (fact_id is None) == (subject_id is None):
        raise ApiError(
            type="invalid_forget_scope",
            message="exactly one of fact_id or subject_id is required",
            field="fact_id" if fact_id is not None else "subject_id",
        )

    if fact_id is not None:
        counters = await _forget_fact(session, project_id=project_id, fact_id=fact_id, mode=mode)
        scope = "fact"
    else:
        counters = await _forget_subject(
            session, project_id=project_id, subject_id=subject_id, mode=mode
        )
        scope = "subject"

    receipt = ForgetReceipt(
        project_id=project_id,
        scope=scope,
        fact_id=fact_id,
        subject_id=subject_id,
        mode=mode,
        counters=counters,
    )
    session.add(receipt)
    await session.flush()
    return receipt, counters


async def _forget_fact(
    session: AsyncSession, *, project_id: str, fact_id: uuid.UUID, mode: str
) -> dict[str, int]:
    fact = await get_fact(session, fact_id)
    if fact.project_id != project_id:
        # Same 404 as an unknown id: a fact never leaks outside its project.
        raise ApiError(
            type="fact_not_found",
            message=f"Fact {fact_id} does not exist",
            field="fact_id",
            status_code=404,
        )
    target = FactStatus.disabled if mode == "disable" else FactStatus.deleted
    if fact.status is target:
        # Idempotent: forget an already-disabled or already-deleted fact is a
        # no-op, not an error (B3: disabled->disabled and deleted->deleted
        # were raising 500 before the matrix was widened).
        key = "facts_disabled" if mode == "disable" else "facts_deleted"
        return {key: 0}
    await transition_fact_status(session, fact_id, target)
    key = "facts_disabled" if mode == "disable" else "facts_deleted"
    return {key: 1}


async def _forget_subject(
    session: AsyncSession, *, project_id: str, subject_id: str, mode: str
) -> dict[str, int]:
    if mode == "disable":
        facts = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.project_id == project_id,
                        Fact.subject_id == subject_id,
                        Fact.status.in_([FactStatus.active, FactStatus.candidate]),
                    )
                )
            )
            .scalars()
            .all()
        )
        for fact in facts:
            await transition_fact_status(session, fact.id, FactStatus.disabled)
        return {"facts_disabled": len(facts)}

    # mode == "delete": real erasure of everything the subject has in this
    # project. The facts' embeddings leave with the rows; the self-FK
    # (supersedes_id) is satisfied within the single DELETE statement.
    # Episode chunks follow the events via ON DELETE CASCADE (migration
    # 0027); feedback rows pointing at the subject's facts are deleted
    # FIRST (free-text comments may carry personal data, and the fact link
    # is what scopes them to this subject -- 0033's SET NULL is the backstop
    # for any straggler, not the erasure path).
    feedback_deleted = (
        await session.execute(
            delete(Feedback).where(
                Feedback.project_id == project_id,
                Feedback.fact_id.in_(
                    select(Fact.id).where(
                        Fact.project_id == project_id, Fact.subject_id == subject_id
                    )
                ),
            )
        )
    ).rowcount
    facts_deleted = (
        await session.execute(
            delete(Fact).where(
                Fact.project_id == project_id, Fact.subject_id == subject_id
            )
        )
    ).rowcount
    conflicts_deleted = (
        await session.execute(
            delete(ConflictSet).where(
                ConflictSet.project_id == project_id,
                ConflictSet.subject_id == subject_id,
            )
        )
    ).rowcount
    events_deleted = (
        await session.execute(
            delete(Event).where(
                Event.project_id == project_id, Event.subject_id == subject_id
            )
        )
    ).rowcount
    traces_deleted = (
        await session.execute(
            delete(ContextTrace).where(
                ContextTrace.project_id == project_id,
                ContextTrace.subject_id == subject_id,
            )
        )
    ).rowcount
    # Learned predicate synonyms (B5a): scoped per subject, derived from the
    # facts just erased. Keeping them would deterministically hijack future
    # candidates toward a canonical predicate with no fact left to justify
    # it -- including a false positive frozen forever by first-discovery-wins.
    predicate_aliases_deleted = (
        await session.execute(
            delete(PredicateAlias).where(
                PredicateAlias.project_id == project_id,
                PredicateAlias.subject_id == subject_id,
            )
        )
    ).rowcount
    # Channel-identity mappings resolving TO the erased subject would dangle
    # (every lookup through them targets a scope that no longer exists).
    # Merge tombstones keyed ON the subject id (alias_value) are history, not
    # live routing, and are left alone.
    subject_aliases_deleted = (
        await session.execute(
            delete(SubjectAlias).where(
                SubjectAlias.project_id == project_id,
                SubjectAlias.canonical_subject_id == subject_id,
            )
        )
    ).rowcount
    return {
        "facts_deleted": facts_deleted,
        "conflict_sets_deleted": conflicts_deleted,
        "events_deleted": events_deleted,
        "traces_deleted": traces_deleted,
        "feedback_deleted": feedback_deleted,
        "predicate_aliases_deleted": predicate_aliases_deleted,
        "subject_aliases_deleted": subject_aliases_deleted,
    }
