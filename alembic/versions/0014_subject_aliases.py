"""feat: identity resolution — subject_aliases + subject merge receipts

Revision ID: 0014_subject_aliases
Revises: 0013_fact_volatility
Create Date: 2026-08-10

- `subject_aliases`: N channel identifiers (telegram, email, device...) ->
  1 canonical subject, per project. Uniqueness of (project_id, alias_kind,
  alias_value) is enforced by the database: two concurrent registrations of
  the same alias can never diverge. `ix_subject_aliases_lookup` serves the
  fragmentation detector (lookup by value, across all kinds);
  `ix_subject_aliases_canonical` serves the re-pointing done by a merge.
- `subject_merge_receipts`: timestamped receipt of each source->target
  merge (same philosophy as forget_receipts); `counters` says what moved,
  `moved` journals the exact ids per table — the information a future
  guarded un-merge would need (see app/ledger/subjects.py).
- RLS ENABLE + FORCE + the same policy as 0006 on BOTH tables (decision:
  the historical forget_receipts table isn't under RLS, but nothing
  justifies reproducing that gap on new tables that carry subject
  identities). No GRANT needed: the ALTER DEFAULT PRIVILEGES from 0006
  covers tables created by the migration role.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0014_subject_aliases"
down_revision: str | None = "0013_fact_volatility"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_RLS_TABLES = ("subject_aliases", "subject_merge_receipts")

# Same expression as migration 0006 (NULLIF: '' after a reverted SET LOCAL
# on a pooled connection must mean "no context", exactly like NULL).
POLICY = """
    USING (
        NULLIF(current_setting('haki.project_id', true), '') IS NULL
        OR project_id = current_setting('haki.project_id', true)
    )
"""


def upgrade() -> None:
    op.create_table(
        "subject_aliases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("alias_kind", sa.String(64), nullable=False),
        sa.Column("alias_value", sa.String(256), nullable=False),
        sa.Column("canonical_subject_id", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "project_id", "alias_kind", "alias_value",
            name="uq_subject_aliases_identity",
        ),
    )
    op.create_index(
        "ix_subject_aliases_lookup", "subject_aliases", ["project_id", "alias_value"]
    )
    op.create_index(
        "ix_subject_aliases_canonical",
        "subject_aliases",
        ["project_id", "canonical_subject_id"],
    )

    op.create_table(
        "subject_merge_receipts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("source_subject_id", sa.String(128), nullable=False),
        sa.Column("target_subject_id", sa.String(128), nullable=False),
        sa.Column("counters", JSONB(), nullable=False, server_default="{}"),
        sa.Column("moved", JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_subject_merge_receipts_project",
        "subject_merge_receipts",
        ["project_id", "created_at"],
    )

    for table in NEW_RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY haki_project_isolation ON {table} {POLICY}")


def downgrade() -> None:
    for table in NEW_RLS_TABLES:
        op.execute(f"DROP POLICY IF EXISTS haki_project_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_subject_merge_receipts_project", table_name="subject_merge_receipts")
    op.drop_table("subject_merge_receipts")
    op.drop_index("ix_subject_aliases_canonical", table_name="subject_aliases")
    op.drop_index("ix_subject_aliases_lookup", table_name="subject_aliases")
    op.drop_table("subject_aliases")
