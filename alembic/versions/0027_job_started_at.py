"""Atomic job claiming: the timestamp a claim is made at

Revision ID: 0027_job_started_at
Revises: 0026_fact_observed_at
Create Date: 2026-08-22

`JobStatus.running` has existed since the jobs table did and was never
assigned to anything. Nothing used `FOR UPDATE SKIP LOCKED` either, so
every caller selected `pending` rows and processed them: two workers, or
one worker and one POST /v1/consolidate, picked up the same job and
extracted the same events twice.

The per-subject advisory lock serialised the writes, so nothing corrupted.
But the second extraction is a second LLM call, non-deterministic even at
temperature 0 -- it can produce a slightly different value and open a
conflict set against the fact the first pass had just written. To a
customer that looks like the memory arguing with itself.

`started_at` is when a worker claimed the job. It is what makes a claim
reclaimable: a worker killed mid-job (deploy, OOM, SIGKILL) never marks it
failed, so without a reclaim window the job would sit in `running` for
ever -- a stuck-job bug in place of a duplicate-processing one. See
STALE_CLAIM_AFTER in app.consolidator.

Existing pending/failed rows are unaffected: NULL started_at, still
claimable. No row is currently `running` (nothing ever set it), so there
is nothing to reconcile.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0027_job_started_at"
down_revision: str | None = "0026_fact_observed_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN started_at timestamptz")
    # The claim query's own filter: kind + status, oldest first.
    op.execute(
        "CREATE INDEX ix_jobs_claimable ON jobs (kind, status, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_jobs_claimable")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS started_at")
