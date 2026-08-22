"""Switch search_vector (facts + events) from the 'simple' config to 'english'

Revision ID: 0023_search_vector_english
Revises: 0022_forget_receipts_rls
Create Date: 2026-08-20

Found via external code audit (3rd report, claim 2.1), confirmed by direct
reading: `websearch_to_tsquery` ANDs every term of the query together, and
PostgreSQL's 'simple' config has NO stopword list at all -- so a natural
language question like "what is Sarah's dog's name" becomes a query that
literally requires "what" AND "is" AND "the" to be present in the indexed
text. `search_text`/`index_text` (migrations 0004, 0019) are compact
"predicate value" / episode strings -- they almost never contain these
connecting words. Measured result: the full-text axis (weight 0.25 of the
hybrid score, see the docstring of app/context/__init__.py) almost never
matched on a natural question, neither for facts nor for episodes (the
same ts_query is reused for both, see mechanism E1a).

'english' fixes the problem on the query side (stopwords dropped from the
AND) and stems both sides (the generated column AND websearch_to_tsquery,
see app/context/__init__.py) -- both must change together or the lexemes
stop matching and the axis drops to zero matches, which is strictly worse
than the original bug.

GENERATED column: Postgres does not allow ALTER ... SET EXPRESSION, so it
must be DROP then re-ADD (rewrites the column on every existing row --
accepted here, dev/eval-sized table).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0023_search_vector_english"
down_revision: str | None = "0022_forget_receipts_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_search_vector")
    op.execute("ALTER TABLE facts DROP COLUMN search_vector")
    op.execute(
        "ALTER TABLE facts ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(search_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_facts_search_vector ON facts USING gin (search_vector)")

    op.execute("DROP INDEX IF EXISTS ix_events_search_vector")
    op.execute("ALTER TABLE events DROP COLUMN search_vector")
    op.execute(
        "ALTER TABLE events ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(index_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_events_search_vector ON events USING gin (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_search_vector")
    op.execute("ALTER TABLE facts DROP COLUMN search_vector")
    op.execute(
        "ALTER TABLE facts ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', coalesce(search_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_facts_search_vector ON facts USING gin (search_vector)")

    op.execute("DROP INDEX IF EXISTS ix_events_search_vector")
    op.execute("ALTER TABLE events DROP COLUMN search_vector")
    op.execute(
        "ALTER TABLE events ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', coalesce(index_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_events_search_vector ON events USING gin (search_vector)")
