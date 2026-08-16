"""Reclassification safety net: reclassified_at on facts

Revision ID: 0021_reclassified_at
Revises: 0020_temporal_range
Create Date: 2026-08-16

Found by code review (16 aout): mechanism C's automatic overflow
reclassification (app.consolidator._apply_candidate, a 3rd competing
"state" value flipping the whole identity to "event") activates and
serves all 3 facts with no safety net, assuming "a genuine scalar never
reaches a 3rd competing value". Non-deterministic extraction can defeat
that assumption -- 3 "create" candidates for what is actually one scalar
attribute (e.g. employer) would then serve 3 contradictory values as
equally trusted.

Rather than a new calibrated threshold (which the reclassification was
explicitly designed to avoid) or reintroducing the quarantine this
mechanism replaces, `reclassified_at` (nullable, NULL = never
reclassified) follows the project's existing honest-degradation
convention: never hidden, always marked (contested facts,
"unconfirmed"/"stale" freshness -- see app/context/__init__.py). Surfaced
in the served packet and the SDK-rendered prompt text.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021_reclassified_at"
down_revision: str | None = "0020_temporal_range"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("reclassified_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("facts", "reclassified_at")
