"""sprint 10: memoire episodique (events.embedding 384 dims)

Revision ID: 0007_episodic_events
Revises: 0006_auth_feedback_rls
Create Date: 2026-08-02

Sprint 10 (harnais d'evaluation) : les questions « que s'est-il passe /
quand » ne trouvaient aucune reponse — l'extracteur ne garde que les faits
durables et jette les evenements dates (0% LoCoMo au premier run). Le
ContextPacket sert maintenant aussi des EPISODES : les evenements sources
les plus proches de la requete, avec leur date.

- events.embedding vector(384) : rempli par le consolidator (embedder local,
  texte = kind + payload tronque). Derive, re-computable — pas une
  modification metier de l'evenement (append-only preserve sur le contenu).
- index hnsw cosinus, meme pattern que facts (migration 0003).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_episodic_events"
down_revision: str | None = "0006_auth_feedback_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN embedding vector(384)")
    op.execute(
        "CREATE INDEX ix_events_embedding_hnsw ON events "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_events_embedding_hnsw")
    op.execute("ALTER TABLE events DROP COLUMN embedding")
