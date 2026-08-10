"""fix: duplicate active facts — repair, reinforcement, partial unique index

Revision ID: 0012_facts_active_unique
Revises: 0011_context_trace_timing
Create Date: 2026-08-10

Three things, in this order (the order matters):

1. Repair pre-existing active duplicates: the Consolidator's TOCTOU race
   (an application-level duplicate check with no lock, fixed in
   app/consolidator alongside this migration) could leave several `active`
   facts for the same (project_id, subject_id, predicate). The most recent
   one (recorded_from) is kept, the rest become `superseded` (version
   incremented, same as transition_fact_status). supersedes_id is NOT set:
   a race duplicate is not a replacement lineage.
2. Reinforcement columns: a NEW event re-asserting the SAME value updates
   the existing fact (counter + date) instead of creating a row.
3. Partial unique index on (project_id, subject_id, predicate) WHERE
   status = 'active': a DB backstop — two active facts with the exact same
   predicate become impossible regardless of application code. SEMANTIC
   dedup (different predicates, same concept) stays covered by the
   consolidator's advisory lock, not by this index.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_facts_active_unique"
down_revision: str | None = "0011_context_trace_timing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1) Repair pre-existing active duplicates (TOCTOU leftovers): keep the
    # most recent per (project_id, subject_id, predicate), supersede the rest.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY project_id, subject_id, predicate
                       ORDER BY recorded_from DESC, id DESC
                   ) AS rn
            FROM facts
            WHERE status = 'active'
        )
        UPDATE facts
        SET status = 'superseded', version = version + 1
        FROM ranked
        WHERE facts.id = ranked.id AND ranked.rn > 1
        """
    )

    # 2) Write-time reinforcement metadata.
    op.add_column(
        "facts",
        sa.Column(
            "reinforcement_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "facts", sa.Column("last_reinforced_at", sa.DateTime(timezone=True))
    )

    # 3) DB backstop: at most one ACTIVE fact per subject+predicate.
    op.create_index(
        "uq_facts_active_subject_predicate",
        "facts",
        ["project_id", "subject_id", "predicate"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    # The data repair of upgrade() is NOT reversed: the superseded
    # duplicates were never a state worth restoring.
    op.drop_index("uq_facts_active_subject_predicate", table_name="facts")
    op.drop_column("facts", "last_reinforced_at")
    op.drop_column("facts", "reinforcement_count")
