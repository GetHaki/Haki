import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Origin-trust levels (M8 — provenance as authority). Declared by the
# AUTHENTICATED caller (the developer's backend) or derived server-side from
# actor_type — never by the model: no LLM-facing surface accepts this field
# as a parameter. Haki cannot DETECT where content came from; it enforces
# the consequences of what the caller honestly declares:
#   trusted      direct message from the tracked subject (default)
#   semi_trusted output of the agent/tooling itself (MCP haki_capture,
#                any event whose actor_type says "agent"/"tool"/"system")
#   third_party  a third participant in the subject's conversation (group
#                chat): facts are attributed to them, never to the subject
#   untrusted    ingested external content (document, web page, forwarded
#                text): never served, never auto-activated — see consolidator
ORIGIN_TRUST_LEVELS: tuple[str, ...] = (
    "trusted",
    "semi_trusted",
    "third_party",
    "untrusted",
)

# Authority ordering used by the consolidator: a candidate born from a
# strictly lower-ranked event never displaces a fact born from a higher-
# ranked one. Equality is allowed on purpose (an MCP-captured fact must
# keep superseding earlier MCP-captured facts).
ORIGIN_TRUST_RANK: dict[str, int] = {
    "untrusted": 0,
    "third_party": 1,
    "semi_trusted": 2,
    "trusted": 3,
}

# actor_type values that make an event "the agent talking", not the human.
AGENT_ACTOR_TYPES: frozenset[str] = frozenset({"agent", "tool", "system"})


class Event(Base):
    """Source event (contract B.1). Append-only: business content is never
    UPDATEd. The only tolerated writes after insert are the derived
    retrieval fields below (embedding, index_text) -- both re-computable
    from kind+payload (+ extracted facts for index_text), set once by the
    consolidator.

    Bitemporal: occurred_at is business time, recorded_at is system time.
    """

    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_events_idempotency"),
        Index("ix_events_timeline", "project_id", "subject_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # Scope
    org_id: Mapped[str] = mapped_column(String(128))
    project_id: Mapped[str] = mapped_column(String(128))
    subject_type: Mapped[str] = mapped_column(String(64), default="user")
    subject_id: Mapped[str] = mapped_column(String(128))

    # Provenance
    actor_type: Mapped[str | None] = mapped_column(String(64))
    actor_id: Mapped[str | None] = mapped_column(String(128))
    agent_id: Mapped[str | None] = mapped_column(String(128))
    thread_id: Mapped[str | None] = mapped_column(String(128))
    run_id: Mapped[str | None] = mapped_column(String(128))

    kind: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    payload: Mapped[dict] = mapped_column(JSONB)
    source: Mapped[dict | None] = mapped_column(JSONB)
    classification: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    retention_policy: Mapped[str | None] = mapped_column(String(128))

    hash: Mapped[str] = mapped_column(String(71))
    idempotency_key: Mapped[str] = mapped_column(String(256))

    # Episodic retrieval (sprint 10): derived embedding of kind + truncated
    # payload, set once by the consolidator. NULL until consolidated. Once
    # this event's own facts are known (mechanism E3, 15 aout, migration
    # 0022), re-derived from index_text instead so the embedding also
    # reflects what was extracted, not just the raw payload.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))

    # True key merging (mechanism E3, migration 0022): kind + truncated
    # payload, concatenated with the predicate/value of every fact
    # extracted from THIS event (app.consolidator, via Fact.source_event_ids)
    # -- the compressed signal folded into the raw key at INDEX time,
    # rather than merged with facts only at read time (the "unified pool"/
    # key-merging-at-read mechanism already in app.context, 13 aout, which
    # this does not replace: that fuses facts and episodes by RANK at query
    # time; this changes what a single episode's own index CONTAINS).
    # Indexing-only field: the packet's episode excerpt is still rendered
    # fresh from kind+payload (app.context.episode_excerpt), never from
    # this column, so a concatenated fact can never leak into what the
    # agent reads as a verbatim source quote.
    index_text: Mapped[str | None] = mapped_column(String)
    # Precomputed tsvector of index_text (generated column, mechanism E1a,
    # migration 0022): same pattern as facts.search_vector (migration
    # 0004) -- an episode can now be found by an exact lexical match (a
    # name, an identifier) even when it is not the closest embedding
    # neighbour, the same axis facts already had and episodes did not.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(index_text, ''))", persisted=True),
    )

    # Origin trust (M8): declared by the authenticated caller or derived
    # from actor_type at write time (ledger.write_events) — see
    # ORIGIN_TRUST_LEVELS above. server_default keeps pre-existing rows on
    # the implicit full-authority behavior they always had.
    origin_trust: Mapped[str] = mapped_column(
        String(16), default="trusted", server_default="trusted"
    )
