"""Mechanism F1: temporal_range (ISO range) on facts

Revision ID: 0020_temporal_range
Revises: 0019_episode_index_text
Create Date: 2026-08-15

Book, Part 3.6 / 4.4: a relative time expression ("last week", "a few
days ago") overwritten by the message's own timestamp at extraction is
information destroyed, never recoverable. `temporal_range` captures the
ISO range resolved by the extractor, anchored on the source event's
`occurred_at` -- deliberately distinct from `valid_from` (always the
MESSAGE's date, never the DESCRIBED event's). Nullable, no existing row
is affected: a fact with no relative time expression (an absolute date
already in `value`, or no time reference at all) keeps
`temporal_range = NULL`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020_temporal_range"
down_revision: str | None = "0019_episode_index_text"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("temporal_range", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("facts", "temporal_range")
