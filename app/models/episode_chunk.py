import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.models.base import Base


class EpisodeChunk(Base):
    """One retrievable, servable slice of an event's payload.

    Derived data, never a second source of truth: `events` remains the
    append-only ledger, and every row here is reconstructible from its
    parent event by `app.context.chunking.chunk_payload` (see
    scripts/backfill_episode_chunks.py). Dropping and rebuilding this whole
    table costs an embedding pass and loses nothing.

    Why it exists: an event was previously indexed, embedded and served
    whole. On the eval corpus that made the median episode cost 810 tokens
    of a 900-token budget, truncated a quarter of the corpus at 4 000
    characters, and left 87.6 % of each episode outside the embedder's
    ~128-token window (migration 0024 for the measurements).

    Denormalised columns (project_id, subject_id, occurred_at,
    origin_trust) are copied from the parent event on purpose. Retrieval
    filters on all four before ranking, and going through a join would
    keep Postgres from using the hnsw and GIN indexes on this table --
    the whole point of the two-CTE candidate generation.

    Three of the four cannot drift: events are append-only, and
    project_id, occurred_at and origin_trust are never rewritten. The
    fourth, `subject_id`, IS rewritten -- by a subject merge, the one
    operation that moves a subject's history under another id -- so
    `app.ledger.subjects.merge_subjects` updates this table alongside
    events, facts, conflicts and traces. Anything that gains the right to
    rewrite one of these four has to do the same.
    """

    __tablename__ = "episode_chunks"
    __table_args__ = (
        # One row per (event, position). Makes re-chunking an event an
        # upsert rather than a duplicate, and gives the backfill a natural
        # idempotency key.
        UniqueConstraint("event_id", "ordinal", name="uq_episode_chunks_event_ordinal"),
        # Scope + point-of-view filter, in the order the retrieval query
        # applies them.
        Index("ix_episode_chunks_scope", "project_id", "subject_id", "occurred_at"),
        # Ordered neighbour lookup for the context window (mechanism F2).
        Index("ix_episode_chunks_event_ordinal", "event_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)

    project_id: Mapped[str] = mapped_column(String(128))
    subject_id: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    origin_trust: Mapped[str] = mapped_column(String(16), default="trusted")

    # What is SERVED: a verbatim slice of the payload, nothing else. Kept
    # separate from index_text so that a fact folded into the index can
    # never leak into what the agent reads as a direct source quote.
    text: Mapped[str] = mapped_column(String)
    # What is MATCHED: the text, plus (once the fact-to-chunk link lands)
    # the facts extracted from this slice -- key merging at index time,
    # LongMemEval's K = V + fact. Equal to `text` until then.
    index_text: Mapped[str] = mapped_column(String)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        # Same configuration as facts.search_vector and events.search_vector
        # -- see the note on those columns; app.db.verify_fts_config
        # enforces it against settings.fts_config at startup.
        Computed(
            f"to_tsvector('{settings.fts_config}', coalesce(index_text, ''))",
            persisted=True,
        ),
    )
