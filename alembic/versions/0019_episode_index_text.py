"""Mechanisms E1a/E3 (15 aout, Sprint 1): index_text + tsvector on events

Revision ID: 0019_episode_index_text
Revises: 0018_memory_form
Create Date: 2026-08-15

Two gaps confirmed by reading primary sources on August 14 (see
research/Diagnostic_Couverture_2026-08-14.md and the same day's handoff):
the embedder truncates at ~100-130 words, and the "unified pool" shipped
12-13 aout is rank merging (fusion at read time), not the true key
merging from the LongMemEval paper (fusion at INDEX time, +9.4%
recall/+5.4% accuracy measured in that paper).

E1a: `events.index_text` gets the same full-text treatment `facts.
search_text` already had (migration 0004) -- generated `search_vector`
column + GIN index, the exact same pattern. Without this, an episode
could only ever be found by semantic similarity; an episode sharing an
exact keyword with the query (a proper noun, an identifier) but weak
vector proximity stayed invisible to the mechanism facts already had.

E3: `index_text` (not `embedding` alone) becomes the target of true key
merging -- an episode's raw text CONCATENATED with the facts extracted
from it, indexed once at write time (app.consolidator), instead of only
merged at read time (the "unified pool" from 13 aout, which stays in
place for RANKING facts and episodes together -- this project does not
replace it, it enriches what each episode's own index CONTAINS).
`index_text` is an indexing-only field: the verbatim excerpt served in
the packet (`episode_excerpt`) is still computed on the fly from
kind+payload, never from `index_text` -- no risk of a concatenated fact
leaking into what the agent reads as a direct source quote.

server_default NULL: no existing row changes behavior until the
consolidator has repopulated it (events already consolidated before this
migration keep `index_text` NULL -- treated as an empty string by
`to_tsvector`, same GIN index but no full-text term until a new event for
the same subject triggers another consolidation pass over it).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0019_episode_index_text"
down_revision: str | None = "0018_memory_form"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE events ADD COLUMN index_text VARCHAR")
    op.execute(
        "ALTER TABLE events ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', coalesce(index_text, ''))) STORED"
    )
    op.execute(
        "CREATE INDEX ix_events_search_vector ON events USING gin (search_vector)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_events_search_vector")
    op.execute("ALTER TABLE events DROP COLUMN search_vector")
    op.execute("ALTER TABLE events DROP COLUMN index_text")
