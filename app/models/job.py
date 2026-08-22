import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Job(Base):
    """Async work unit.

    Claimed atomically before processing (22 aout): `JobStatus.running`
    existed from the start and was never assigned, and nothing used
    `FOR UPDATE SKIP LOCKED`, so two workers -- or one worker and one
    POST /v1/consolidate -- selected the same pending job and extracted the
    same events twice. The per-subject advisory lock serialised them, but
    the second extraction is a second LLM call: non-deterministic even at
    temperature 0, so it could produce a slightly different value and open
    a conflict set against the fact the first pass had just written. See
    `claim_jobs` in app.consolidator.
    """

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.pending
    )
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # When a worker CLAIMED this job (migration 0027). Two things depend on
    # it: a claimed job is invisible to every other worker, and a job whose
    # claim is older than STALE_CLAIM_AFTER is reclaimable -- otherwise a
    # worker killed mid-job (a deploy, an OOM) would leave it `running`
    # forever, which trades a duplicate-processing bug for a stuck-job bug.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
