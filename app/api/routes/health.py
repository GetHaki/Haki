from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import metrics
from app.db import get_session

router = APIRouter()


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    await session.execute(text("SELECT 1"))
    # `counters`: simple in-memory visibility into the noisy-failure
    # contract (context.ok/degraded, gateway.memory.*, mcp.context.*,
    # consolidator.job.*) — see app/metrics.py. Not Prometheus, process-
    # local, reset on restart.
    return {"status": "ok", "database": "up", "counters": metrics.snapshot()}
