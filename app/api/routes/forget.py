"""POST /v1/forget — minimal erasure endpoint (sprint 4).

Disable (reversible, via the Ledger lifecycle) or delete (real erasure)
one fact or one whole subject within a project. Every call is journaled
in `forget_receipts` and the response carries the receipt id plus the
counters of what actually happened.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger, policy
from app.db import get_session
from app.schemas import ForgetRequest, ForgetResponse

router = APIRouter()


@router.post("/forget", response_model=ForgetResponse)
async def forget_endpoint(
    request: ForgetRequest, session: AsyncSession = Depends(get_session)
) -> ForgetResponse:
    receipt, counters = await ledger.forget(
        session,
        project_id=request.project_id,
        subject_id=request.subject_id,
        fact_id=request.fact_id,
        mode=request.mode,
    )
    # Policy Engine: every forget is journaled (US 42 — audit).
    policy.audit_forget(
        project_id=request.project_id, mode=request.mode, scope=receipt.scope
    )
    await session.commit()
    return ForgetResponse(
        mode=request.mode,
        scope=receipt.scope,
        forget_id=receipt.id,
        **counters,
    )
