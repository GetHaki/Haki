"""0036: RLS off on non-project tables that never had a policy.

Found live (2026-09-11): ROW LEVEL SECURITY is enabled with ZERO policies
on credit_transactions (and feedback, jobs, organization_invites,
organization_members, embedding_space) — almost certainly auto-enabled by
the Supabase dashboard's table creator. With no policy, Postgres
default-denies: haki_app's INSERTs fail outright ("new row violates
row-level security policy") and its SELECTs silently return zero rows.
Concretely broken in production: every credit-ledger write (webhook
grants, capture debits) and every ledger read (GET /v1/billing/credits
transactions) for the credit_transactions table.

These tables are not project-scoped — 0010 documents credit_transactions
as intentionally RLS-free ("comme organizations et api_keys"), and 0011 /
0017 / 0020 / 0025 / 0031 all DISABLE ROW LEVEL SECURITY on the same
class of tables. This migration extends that same rule to every table
found with RLS-on-but-policyless. NO FORCE first: even a table owner must
not be able to silently re-enable a policy-less RLS later.

Project tables WITH a real haki_project_isolation policy (events, facts,
...) are untouched.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0036_no_rls_billing_tables"
down_revision: str | None = "0035_dodo_subscription_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NO_RLS_TABLES = (
    "credit_transactions",
    "feedback",
    "jobs",
    "organization_invites",
    "organization_members",
    "embedding_space",
)


def upgrade() -> None:
    for table in NO_RLS_TABLES:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    # No-op: re-enabling RLS with no policy would just reintroduce the
    # exact bug this migration fixes. If real RLS is ever wanted here,
    # it needs an actual policy authored alongside enabling it.
    pass
