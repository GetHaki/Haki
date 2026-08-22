"""Episodic memory: one retrievable unit per turn, not per event

Revision ID: 0024_episode_chunks
Revises: 0023_search_vector_english
Create Date: 2026-08-21

An event was indexed, embedded and served whole. Measured on the 272 real
LoCoMo sessions of the eval corpus, that meant a median episode cost 810
tokens of the eval's 900-token budget (so one episode at a time, and
nothing left for facts), 25.4 % of sessions were truncated at
EPISODE_TEXT_CHARS = 4 000 (7.1 % of the corpus destroyed outright), and
**0 of 272 episodes were fully embedded** -- the local embedder truncates
at ~128 tokens (verified directly against LocalEmbedder in this session:
two texts differing only after that point score cosine similarity
0.9999999999999999), so the median episode had 12.4 % of itself in the
index.

Cutting on turn boundaries, everything else held constant, on the 1 536
non-adversarial LoCoMo questions (gold evidence served under a 900-token
budget): claimed 19.0 % -> 66.3 % by the external audit this migration
implements -- not re-run against this project's own bench, but the
mechanism (a turn is small enough to be embedded whole: 5 882/5 882 real
LoCoMo turns are covered by a 128-token window, against 0/272 sessions)
is independently verified above.

Derived, not authoritative
---------------------------
`events` stays the append-only ledger and is not touched. Every row here
is reconstructible from its parent event by
`app.context.chunking.chunk_payload`; dropping and rebuilding the table
costs an embedding pass and loses nothing. ON DELETE CASCADE, so a forget
that removes an event removes its chunks with it -- the raw text lives
here too, and leaving it behind would defeat the receipt.

Denormalised project_id / subject_id / occurred_at / origin_trust: the
retrieval query filters on all four before ranking, and a join would keep
Postgres from using the hnsw and GIN indexes this table exists for. Three
of the four cannot drift (events are append-only and none of project_id/
occurred_at/origin_trust is ever rewritten); subject_id CAN, by
app.ledger.subjects.merge_subjects, which updates this table alongside
events/facts/conflicts/traces for exactly that reason.

RLS: same policy as every other tenant table in this project
(migrations 0006, 0014, 0022 -- NULLIF('', NULL) expression). This table
holds verbatim user content, so it gets the isolation before it gets a
single row, not retrofitted later the way forget_receipts (0005) was and
0022 had to fix.

Backfill
--------
This migration creates an EMPTY table. Existing events keep working
through nothing -- the episode channel returns no chunk for them until
`scripts/backfill_episode_chunks.py` has run (no LLM call, local embedder
only). Run it once per environment right after the upgrade.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0024_episode_chunks"
down_revision: str | None = "0023_search_vector_english"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same expression as migrations 0006/0014/0022 (NULLIF: '' after a
# reverted SET LOCAL on a pooled connection must mean "no context",
# exactly like NULL).
POLICY = """
    USING (
        NULLIF(current_setting('haki.project_id', true), '') IS NULL
        OR project_id = current_setting('haki.project_id', true)
    )
"""


# Read from the same single source of truth the lexical axis uses
# (app.context.fts / app.db.verify_fts_config), so this column cannot be
# created with a different configuration from the two it has to be
# comparable with (facts.search_vector, events.search_vector).
def _fts_config() -> str:
    from app.config import settings

    return settings.fts_config


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE episode_chunks (
            id uuid PRIMARY KEY,
            event_id uuid NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            ordinal integer NOT NULL,
            project_id varchar(128) NOT NULL,
            subject_id varchar(128) NOT NULL,
            occurred_at timestamptz NOT NULL,
            origin_trust varchar(16) NOT NULL DEFAULT 'trusted',
            text varchar NOT NULL,
            index_text varchar NOT NULL,
            embedding vector(384),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('{_fts_config()}', coalesce(index_text, ''))
            ) STORED
        )
        """
    )
    op.execute(
        "ALTER TABLE episode_chunks ADD CONSTRAINT uq_episode_chunks_event_ordinal "
        "UNIQUE (event_id, ordinal)"
    )
    op.execute(
        "CREATE INDEX ix_episode_chunks_scope ON episode_chunks "
        "(project_id, subject_id, occurred_at)"
    )
    op.execute(
        "CREATE INDEX ix_episode_chunks_event_ordinal ON episode_chunks (event_id, ordinal)"
    )
    op.execute(
        "CREATE INDEX ix_episode_chunks_embedding_hnsw ON episode_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_episode_chunks_search_vector ON episode_chunks "
        "USING gin (search_vector)"
    )
    op.execute("ALTER TABLE episode_chunks ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE episode_chunks FORCE ROW LEVEL SECURITY")
    op.execute("CREATE POLICY haki_project_isolation ON episode_chunks " + POLICY)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS episode_chunks")
