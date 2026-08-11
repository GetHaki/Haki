"""feat: provenance as authority — origin_trust on events and facts

Revision ID: 0015_provenance_trust
Revises: 0014_subject_aliases
Create Date: 2026-08-10

Each event carries an origin-trust level (trusted / semi_trusted /
third_party / untrusted), declared by the authenticated backend or derived
from actor_type; each fact inherits the level of its source event. The
consolidator uses it as an authority rule: an untrusted-origin candidate is
never served without human resolution, a strictly lower rank never displaces
a strictly higher one, a third_party fact is attributed to that third party.

Backward compatible: server_default 'trusted' matches the implicit
full-authority behavior every existing row already had. Backfill
semi_trusted from actor_type (agent/tool/system): exactly what the write
path would have derived — actor_type is the only provenance already
recorded. The UPDATEs pass under RLS since the haki_project_isolation
policy (0006) is permissive without a GUC set (the migration connection's
case). String + Pydantic validation at the write path, no Postgres enum:
adding a level later needs no ALTER TYPE.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_provenance_trust"
down_revision: str | None = "0014_subject_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("origin_trust", sa.String(length=16), nullable=False, server_default="trusted"),
    )
    op.add_column(
        "facts",
        sa.Column("origin_trust", sa.String(length=16), nullable=False, server_default="trusted"),
    )
    op.execute(
        "UPDATE events SET origin_trust = 'semi_trusted' "
        "WHERE actor_type IN ('agent', 'tool', 'system')"
    )
    op.execute(
        "UPDATE facts SET origin_trust = 'semi_trusted' "
        "WHERE EXISTS ("
        "  SELECT 1 FROM events e"
        "  WHERE e.id = ANY(facts.source_event_ids)"
        "  AND e.origin_trust = 'semi_trusted'"
        ")"
    )


def downgrade() -> None:
    op.drop_column("facts", "origin_trust")
    op.drop_column("events", "origin_trust")
