"""feat: predicate alias table (fact identity, 11 aout)

Revision ID: 0017_predicate_aliases
Revises: 0016_facts_identity_qualifiers
Create Date: 2026-08-13

`predicate_aliases`: deterministic tier between the exact match and the
semantic fallback in app.consolidator._resolve_existing_fact -- exact
canonical key first, alias table second, semantic fallback last (exact
order from the 11 aout diagnostic: "a fact's identity isn't computed, it's
guessed from a string"). Scoped (project_id, subject_id, alias_predicate):
per subject, not per project -- a project-wide alias would risk a short,
generic predicate string learned for one subject silently hijacking
another subject's genuinely different concept of the same word.
Uniqueness (project_id, subject_id, alias_predicate) enforced by the
database: two concurrent registrations of the same alias can never
diverge -- the consolidator inserts with `ON CONFLICT DO NOTHING`, first
discovery wins.

RLS ENABLE + FORCE + policy identical to 0006/0014 (project isolation). No
GRANT needed: ALTER DEFAULT PRIVILEGES from 0006 covers tables created by
the migration role.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_predicate_aliases"
down_revision: str | None = "0016_facts_identity_qualifiers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same expression as migration 0006/0014 (NULLIF: '' after a reverted SET
# LOCAL on a pooled connection must mean "no context", exactly like NULL).
POLICY = """
    USING (
        NULLIF(current_setting('haki.project_id', true), '') IS NULL
        OR project_id = current_setting('haki.project_id', true)
    )
"""


def upgrade() -> None:
    op.create_table(
        "predicate_aliases",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("alias_predicate", sa.String(128), nullable=False),
        sa.Column("canonical_predicate", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "project_id", "subject_id", "alias_predicate",
            name="uq_predicate_aliases_identity",
        ),
    )
    op.create_index(
        "ix_predicate_aliases_lookup",
        "predicate_aliases",
        ["project_id", "subject_id", "alias_predicate"],
    )

    op.execute("ALTER TABLE predicate_aliases ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE predicate_aliases FORCE ROW LEVEL SECURITY")
    op.execute(f"CREATE POLICY haki_project_isolation ON predicate_aliases {POLICY}")


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS haki_project_isolation ON predicate_aliases")
    op.execute("ALTER TABLE predicate_aliases NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE predicate_aliases DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_predicate_aliases_lookup", table_name="predicate_aliases")
    op.drop_table("predicate_aliases")
