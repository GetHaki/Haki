"""Feedback (sprint 6, extended M10 — "haki_correct"): the single mechanism
behind both POST /v1/feedback (app/api/routes/feedback.py) and the
haki_correct MCP tool (app/mcp_server/__init__.py). Deliberately the SAME
function for both entry points — like `ledger.forget` already is for
POST /v1/forget and the haki_forget MCP tool — so a correction made from
inside the conversation has a byte-identical effect to a direct API call,
never a re-implementation that could drift.

A rating `incorrect` on a fact transitions it to `disputed` through the
Ledger lifecycle: the Context Assembler never serves a disputed fact as
active again (status filter). Every observation is stored in `feedback`
(migration 0006).
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ApiError
from app.ledger.core import ALLOWED_TRANSITIONS, get_fact, transition_fact_status
from app.models import FactStatus, Feedback

VALID_RATINGS = ("useful", "irrelevant", "incorrect")


async def submit_feedback(
    session: AsyncSession,
    *,
    project_id: str,
    rating: str,
    trace_id: uuid.UUID | None = None,
    fact_id: uuid.UUID | None = None,
    comment: str | None = None,
) -> tuple[Feedback, str | None]:
    """Record one quality observation on a trace or a fact.

    Exactly one of trace_id / fact_id is required. Returns (the persisted
    Feedback row, the resulting fact status — None when the feedback
    targeted a trace). Caller commits.
    """
    if rating not in VALID_RATINGS:
        raise ApiError(
            type="invalid_payload",
            message=f"rating must be one of {VALID_RATINGS}",
            field="rating",
        )
    if (trace_id is None) == (fact_id is None):
        raise ApiError(
            type="invalid_payload",
            message="exactly one of trace_id or fact_id is required",
            field="fact_id" if fact_id is not None else "trace_id",
        )

    fact_status: str | None = None
    if fact_id is not None:
        fact = await get_fact(session, fact_id)
        if fact.project_id != project_id:
            # Same 404 as an unknown id: a fact never leaks outside its project.
            raise ApiError(
                type="fact_not_found",
                message=f"Fact {fact_id} does not exist",
                field="fact_id",
                status_code=404,
            )
        if (
            rating == "incorrect"
            and FactStatus.disputed in ALLOWED_TRANSITIONS[fact.status]
        ):
            fact = await transition_fact_status(session, fact.id, FactStatus.disputed)
        fact_status = fact.status.value

    row = Feedback(
        project_id=project_id,
        trace_id=trace_id,
        fact_id=fact_id,
        rating=rating,
        comment=comment,
    )
    session.add(row)
    await session.flush()
    return row, fact_status
