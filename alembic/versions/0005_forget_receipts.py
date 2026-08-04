"""sprint 4: table forget_receipts (recu d'effacement)

Revision ID: 0005_forget_receipts
Revises: 0004_search_vector_gin
Create Date: 2026-08-01

Journalise chaque operation d'oubli (`POST /v1/forget` / outil MCP
`haki_forget`) : cible (scope fact|subject), mode (disable|delete) et
compteurs de ce qui a ete reellement fait. C'est l'embryon du recu
d'effacement du PRD (preuve d'oubli opposable).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005_forget_receipts"
down_revision: str | None = "0004_search_vector_gin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE forget_receipts (
            id UUID PRIMARY KEY,
            project_id VARCHAR(128) NOT NULL,
            scope VARCHAR(16) NOT NULL,
            fact_id UUID,
            subject_id VARCHAR(128),
            mode VARCHAR(16) NOT NULL,
            counters JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_forget_receipts_scope "
        "ON forget_receipts (project_id, subject_id, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_forget_receipts_scope")
    op.execute("DROP TABLE IF EXISTS forget_receipts")
