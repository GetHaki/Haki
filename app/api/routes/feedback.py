"""POST /v1/feedback (sprint 6) — quality observation on a trace or a fact.

A rating `incorrect` on a fact transitions it to `disputed` through the
Ledger lifecycle: the Context Assembler never serves a disputed fact as
active again (status filter). Every observation is stored in `feedback`
(migration 0006).

The actual work happens in `ledger.submit_feedback` (app/ledger/feedback.py)
— the same function the haki_correct MCP tool calls, so a correction made
from the conversation has the exact same effect as calling this endpoint
directly.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger
from app.db import get_session
from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def feedback(
    request: FeedbackRequest, session: AsyncSession = Depends(get_session)
) -> FeedbackResponse:
    row, fact_status = await ledger.submit_feedback(
        session,
        project_id=request.project_id,
        rating=request.rating,
        trace_id=request.trace_id,
        fact_id=request.fact_id,
        comment=request.comment,
    )
    await session.commit()
    return FeedbackResponse(feedback_id=row.id, fact_status=fact_status)
