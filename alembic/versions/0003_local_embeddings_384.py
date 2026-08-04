"""sprint 3: embeddings 384 dims (embedder local fastembed par defaut)

Revision ID: 0003_local_embeddings_384
Revises: 0002_consolidation_context
Create Date: 2026-08-01

Sprint 3 (latence) : l'embedder par defaut devient LOCAL (fastembed,
paraphrase-multilingual-MiniLM-L12-v2, 384 dims) pour sortir l'appel reseau
du hot path de POST /v1/context.

- facts.embedding passe de vector(1536) a vector(384) : la colonne est
  supprimee puis recreee (les embeddings sont re-computables depuis les
  faits, pas de donnee irremplacable) ;
- l'index hnsw est recree sur la nouvelle colonne.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_local_embeddings_384"
down_revision: str | None = "0002_consolidation_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_embedding_hnsw")
    # Pas de cast vector(1536) -> vector(384) : on recree la colonne.
    op.execute("ALTER TABLE facts DROP COLUMN embedding")
    op.execute("ALTER TABLE facts ADD COLUMN embedding vector(384)")
    op.execute(
        "CREATE INDEX ix_facts_embedding_hnsw ON facts "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_embedding_hnsw")
    op.execute("ALTER TABLE facts DROP COLUMN embedding")
    op.execute("ALTER TABLE facts ADD COLUMN embedding vector(1536)")
    op.execute(
        "CREATE INDEX ix_facts_embedding_hnsw ON facts "
        "USING hnsw (embedding vector_cosine_ops)"
    )
