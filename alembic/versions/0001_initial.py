"""initial schema: events, facts, jobs + pgvector extension

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-31

Sprint 1 (PRD semaines 1-2) : Ledger + capture idempotente.

TODO(sprint 2+) — Row-Level Security : les colonnes de scope (org_id,
project_id, subject_id, agent_id) sont deja en place sur toutes les tables.
Activer RLS quand l'auth/principals arrivent, par exemple :

    ALTER TABLE events ENABLE ROW LEVEL SECURITY;
    ALTER TABLE facts ENABLE ROW LEVEL SECURITY;
    CREATE POLICY scope_isolation ON events
        USING (project_id = current_setting('haki.project_id', true));
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

fact_status = postgresql.ENUM(
    "candidate", "active", "superseded", "disputed", "disabled", "deleted",
    name="fact_status",
)
job_status = postgresql.ENUM(
    "pending", "running", "done", "failed",
    name="job_status",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Les enums sont crees implicitement par create_table ci-dessous.
    # Ne PAS appeler fact_status.create() ici : la creation serait emise
    # une seconde fois lors du create_table (DuplicateObjectError).

    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("org_id", sa.String(128), nullable=False),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("subject_type", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("actor_type", sa.String(64), nullable=True),
        sa.Column("actor_id", sa.String(128), nullable=True),
        sa.Column("agent_id", sa.String(128), nullable=True),
        sa.Column("thread_id", sa.String(128), nullable=True),
        sa.Column("run_id", sa.String(128), nullable=True),
        sa.Column("kind", sa.String(128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("source", postgresql.JSONB(), nullable=True),
        sa.Column(
            "classification", postgresql.ARRAY(sa.String()), nullable=False
        ),
        sa.Column("retention_policy", sa.String(128), nullable=True),
        sa.Column("hash", sa.String(71), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.UniqueConstraint(
            "project_id", "idempotency_key", name="uq_events_idempotency"
        ),
    )
    op.create_index(
        "ix_events_timeline",
        "events",
        ["project_id", "subject_id", "occurred_at"],
    )

    op.create_table(
        "facts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("org_id", sa.String(128), nullable=False),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("subject_type", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("agent_id", sa.String(128), nullable=True),
        sa.Column("predicate", sa.String(128), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("qualifiers", postgresql.JSONB(), nullable=False),
        sa.Column("status", fact_status, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "recorded_from",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("recorded_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "supersedes_id", sa.Uuid(), sa.ForeignKey("facts.id"), nullable=True
        ),
        sa.Column("source_event_ids", postgresql.ARRAY(sa.Uuid()), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("facts")
    op.drop_index("ix_events_timeline", table_name="events")
    op.drop_table("events")
    job_status.drop(op.get_bind(), checkfirst=True)
    fact_status.drop(op.get_bind(), checkfirst=True)
