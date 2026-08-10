"""feat: fact typology + volatility classes on facts

Revision ID: 0013_fact_volatility
Revises: 0012_facts_active_unique
Create Date: 2026-08-10

Most facts go stale in silence (the subject moves, changes jobs, and
never says so) — supersession only fires on a contradicting event. Every
fact now carries a volatility class (stable/slow/volatile/ephemeral,
default horizon in config) and a kind (attribute/preference/instruction;
"event" is covered by episodic memory, "task" by the write gate's
transient_state rejection). The freshness clock reuses last_reinforced_at
(migration 0012 — a re-assertion of the same value already refreshes it),
no new date column. Backward-compatible defaults: every existing fact
becomes attribute/stable and keeps its current behavior.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_fact_volatility"
down_revision: str | None = "0012_facts_active_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("fact_kind", sa.String(32), nullable=False, server_default="attribute"),
    )
    op.add_column(
        "facts",
        sa.Column("volatility", sa.String(16), nullable=False, server_default="stable"),
    )
    op.create_check_constraint(
        "ck_facts_fact_kind",
        "facts",
        "fact_kind IN ('attribute', 'preference', 'instruction')",
    )
    op.create_check_constraint(
        "ck_facts_volatility",
        "facts",
        "volatility IN ('stable', 'slow', 'volatile', 'ephemeral')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_facts_volatility", "facts", type_="check")
    op.drop_constraint("ck_facts_fact_kind", "facts", type_="check")
    op.drop_column("facts", "volatility")
    op.drop_column("facts", "fact_kind")
