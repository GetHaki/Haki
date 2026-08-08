"""feat: latency + hit-rate instrumentation on context_traces

Revision ID: 0011_context_trace_timing
Revises: 0010_credits
Create Date: 2026-08-08

Adds what's needed for real stats (Overview + trace detail): total
duration, a breakdown by REAL pipeline stage (embed/retrieval/
multi_hop_expansion/episodes -- not generic stage names lifted from an
unconnected mockup), and the number of facts served (for the hit-rate).
Nullable: existing traces never measured this, so they stay NULL rather
than a misleading zero.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0011_context_trace_timing"
down_revision: str | None = "0010_credits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("context_traces", sa.Column("duration_ms", sa.Integer()))
    op.add_column("context_traces", sa.Column("stage_timings", JSONB()))
    op.add_column("context_traces", sa.Column("fact_count", sa.Integer()))


def downgrade() -> None:
    op.drop_column("context_traces", "fact_count")
    op.drop_column("context_traces", "stage_timings")
    op.drop_column("context_traces", "duration_ms")
