"""sprint 3: colonne tsvector generee + index GIN + index de scope

Revision ID: 0004_search_vector_gin
Revises: 0003_local_embeddings_384
Create Date: 2026-08-01

Optimisation latence `context` (benchmark sprint 3 : p95 ~3,2 s a 10k faits
avant cette migration) :

- `facts.search_vector` : tsvector GENERATED ALWAYS depuis search_text.
  Avant, la requete de scoring appelait to_tsvector() a la volee sur CHAQUE
  ligne a CHAQUE requete (parsing texte integral a chaque fois) ; desormais
  le tsvector est calcule une fois a l'ecriture et ts_rank_cd lit la colonne.
- index GIN sur search_vector (utile pour les pre-filtres plein-texte) ;
- index composite de scope (project_id, subject_id, status) : le filtre dur
  du Context Assembler devient un index scan des qu'il y a plusieurs
  tenants/sujets en base.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_search_vector_gin"
down_revision: str | None = "0003_local_embeddings_384"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE facts ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', coalesce(search_text, ''))) STORED"
    )
    op.execute(
        "CREATE INDEX ix_facts_search_vector ON facts USING gin (search_vector)"
    )
    op.execute(
        "CREATE INDEX ix_facts_scope_status ON facts (project_id, subject_id, status)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_scope_status")
    op.execute("DROP INDEX IF EXISTS ix_facts_search_vector")
    op.execute("ALTER TABLE facts DROP COLUMN search_vector")
