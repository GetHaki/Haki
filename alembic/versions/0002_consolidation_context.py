"""sprint 2: fact embeddings + full-text, conflict_sets, context_traces

Revision ID: 0002_consolidation_context
Revises: 0001_initial
Create Date: 2026-08-01

Semaines 3-4 du PRD : Memory Consolidator + Context Assembler.

- facts.embedding vector(1536) nullable + index hnsw (vector_cosine_ops) ;
  hnsw est disponible depuis pgvector 0.5 et prefere a ivfflat car il ne
  demande ni entrainement ni vacuum particulier a faible volume.
- facts.search_text : colonne texte pre-rendue (predicate + value) pour la
  recherche plein-texte (tsvector a la volee dans la requete de scoring).
- conflict_sets : faits incompatibles en attente de resolution.
- context_traces : audit des decisions du Context Assembler.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_consolidation_context"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Type pgvector brut : SQLAlchemy/Alembic ne connait pas vector(1536).
    op.execute("ALTER TABLE facts ADD COLUMN embedding vector(1536)")
    op.add_column("facts", sa.Column("search_text", sa.Text(), nullable=True))
    op.execute(
        "CREATE INDEX ix_facts_embedding_hnsw ON facts "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "conflict_sets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("fact_ids", postgresql.ARRAY(sa.Uuid()), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_conflict_sets_scope",
        "conflict_sets",
        ["project_id", "subject_id", "status"],
    )

    op.create_table(
        "context_traces",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.String(128), nullable=False),
        sa.Column("subject_id", sa.String(128), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("purpose", sa.String(128), nullable=True),
        sa.Column("packet", postgresql.JSONB(), nullable=False),
        sa.Column("decisions", postgresql.JSONB(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_context_traces_scope",
        "context_traces",
        ["project_id", "subject_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_context_traces_scope", table_name="context_traces")
    op.drop_table("context_traces")
    op.drop_index("ix_conflict_sets_scope", table_name="conflict_sets")
    op.drop_table("conflict_sets")
    op.execute("DROP INDEX IF EXISTS ix_facts_embedding_hnsw")
    op.drop_column("facts", "search_text")
    op.drop_column("facts", "embedding")
