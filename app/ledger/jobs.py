"""Consolidation jobs: creation only.

Processing (LLM extraction, dedupe, supersession, conflicts) lives in
`app/consolidator` since sprint 2; `run_pending_consolidations` is
re-exported from there through `app.ledger` for backward compatibility.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job, JobStatus


async def create_consolidation_job(
    session: AsyncSession, *, project_id: str, event_ids: list[uuid.UUID]
) -> Job:
    job = Job(
        kind="consolidate",
        status=JobStatus.pending,
        payload={"project_id": project_id, "event_ids": [str(e) for e in event_ids]},
    )
    session.add(job)
    await session.flush()
    return job
