"""POST /v1/consolidate — dev/ops endpoint (sprint 3).

Triggers `run_pending_consolidations` synchronously: pending and failed
`consolidate` jobs are processed immediately instead of waiting for a worker
run. Useful for local development, end-to-end tests, self-hosted n8n flows
and the `haki verify` CLI scenario. V1: no rate limiting (documented as a
dev/ops endpoint, to be protected with auth when principals land).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.consolidator import run_pending_consolidations
from app.db import get_session_ops

router = APIRouter()


@router.post("/consolidate")
async def consolidate(session: AsyncSession = Depends(get_session_ops)) -> dict[str, int]:
    # Ops session without RLS context: this dev/ops endpoint processes
    # pending jobs across projects by design (documented in the README).
    processed = await run_pending_consolidations(session)
    await session.commit()
    return {"processed": processed}
