"""fix: missing RLS on forget_receipts (security review, 16 aout)

Revision ID: 0022_forget_receipts_rls
Revises: 0021_reclassified_at
Create Date: 2026-08-16

`forget_receipts` (migration 0005) was created BEFORE RLS existed
(migration 0006) and was never retrofitted -- migration 0014 already
notes this in its own docstring while creating the next tables
("the historical forget_receipts table isn't under RLS, but nothing
justifies repeating that gap on new tables") without ever fixing the
original table. Found by external security review (16 aout): a real gap,
not currently exploitable (no GET route exposes forget_receipts, it is
write-only over HTTP) -- but a genuine defense-in-depth weakness on a
table whose very name carries subject identifiers.

ENABLE + FORCE + the same policy as 0006/0014 (identical NULLIF
expression). No GRANT needed: 0006's ALTER DEFAULT PRIVILEGES already
covers this table (created by the same migration role).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0022_forget_receipts_rls"
down_revision: str | None = "0021_reclassified_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same expression as migrations 0006/0014 (NULLIF: '' after a reverted SET
# LOCAL on a pooled connection must mean "no context", exactly like NULL).
POLICY = """
    USING (
        NULLIF(current_setting('haki.project_id', true), '') IS NULL
        OR project_id = current_setting('haki.project_id', true)
    )
"""


def upgrade() -> None:
    op.execute("ALTER TABLE forget_receipts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE forget_receipts FORCE ROW LEVEL SECURITY")
    op.execute("CREATE POLICY haki_project_isolation ON forget_receipts " + POLICY)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS haki_project_isolation ON forget_receipts")
    op.execute("ALTER TABLE forget_receipts NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE forget_receipts DISABLE ROW LEVEL SECURITY")
