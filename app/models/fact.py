import enum
import uuid
from datetime import datetime

from sqlalchemy import Computed, DateTime, Enum, Float, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from app.config import settings
from app.models.base import Base


class FactStatus(str, enum.Enum):
    candidate = "candidate"
    active = "active"
    superseded = "superseded"
    disputed = "disputed"
    disabled = "disabled"
    deleted = "deleted"


# Fact typology (M2). "event" lives in episodic memory (events.embedding),
# "task"/transient states are rejected at the write gate (transient_state)
# and never become facts -- so the typology on this table is deliberately
# small: durable attribute vs durable preference vs durable operating
# instruction given by the subject (NOT a directive to the agent, which is
# rejected as imperative_directive before ever reaching the ledger).
FACT_KINDS: tuple[str, ...] = ("attribute", "preference", "instruction")

# Volatility classes (M2): how fast a fact goes stale WITHOUT any
# contradicting event. Default horizons live in app.config (settings.
# volatility_horizon_*_days), never hardcoded. "stable" has no horizon --
# the pre-M2 behavior every existing fact keeps.
VOLATILITY_CLASSES: tuple[str, ...] = ("stable", "slow", "volatile", "ephemeral")

# Memory form (mechanism C, 15 aout -- migration 0018): whether a
# (subject, predicate, qualifiers) identity holds ONE current scalar value
# ("state" -- the only behavior that existed before this field: create/
# supersede/conflict against the single active fact) or ACCUMULATES
# independent occurrences ("event" -- every create is its own permanently
# active row, never fused/superseded/put in conflict with the others under
# the same identity). See app.consolidator._apply_candidate for where the
# form is decided/inherited, and the module docstring there for the
# deterministic "conflict overflow -> reclassify as event" mechanism that
# is the only sanctioned way an identity moves from state to event once it
# already has an active fact -- never a single candidate's own say-so.
MEMORY_FORMS: tuple[str, ...] = ("state", "event")


class Fact(Base):
    """Versioned, bitemporal memory fact (contract B.2)."""

    __tablename__ = "facts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # Full scope
    org_id: Mapped[str] = mapped_column(String(128))
    project_id: Mapped[str] = mapped_column(String(128))
    subject_type: Mapped[str] = mapped_column(String(64), default="user")
    subject_id: Mapped[str] = mapped_column(String(128))
    agent_id: Mapped[str | None] = mapped_column(String(128))

    predicate: Mapped[str] = mapped_column(String(128))
    value: Mapped[dict] = mapped_column(JSONB)
    qualifiers: Mapped[dict] = mapped_column(JSONB, default=dict)

    status: Mapped[FactStatus] = mapped_column(
        Enum(FactStatus, name="fact_status"), default=FactStatus.candidate
    )
    confidence: Mapped[float | None] = mapped_column(Float)

    # Bitemporality
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recorded_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    recorded_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facts.id"))
    source_event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(Uuid), default=list
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Write-time reinforcement (migration 0015): a NEW source event that
    # re-asserts the exact same canonical value updates these on the
    # existing active fact instead of creating a row. See app/consolidator
    # (_reinforce_or_count_duplicate) for the rule and why value equality
    # is required (no embedding-distance threshold can separate a
    # rephrasing from a genuine value update — measured, not assumed).
    reinforcement_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    last_reinforced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    # Typology + volatility (M2, migration 0016). The freshness clock reuses
    # last_reinforced_at above (falls back to valid_from, then recorded_from)
    # -- a write-time reinforcement (migration 0015) already refreshes it on
    # every re-assertion of the same value, so no separate "confirmed_at"
    # column is needed.
    fact_kind: Mapped[str] = mapped_column(String(32), default="attribute")
    volatility: Mapped[str] = mapped_column(String(16), default="stable")

    # Memory form (mechanism C, migration 0018) -- see MEMORY_FORMS above.
    memory_form: Mapped[str] = mapped_column(
        String(16), default="state", server_default="state"
    )

    # Set when this fact was activated by the AUTOMATIC overflow
    # reclassification (a 3rd competing "state" value flipping the whole
    # identity to "event", app.consolidator._apply_candidate) rather than
    # an extractor declaring memory_form="event" up front. None otherwise.
    #
    # Safety net added by code review (16 aout): the reclassification trusts
    # a deterministic heuristic ("a genuine scalar never reaches a 3rd
    # competing value") that non-deterministic extraction can defeat --
    # three "create" candidates for what is actually one scalar attribute
    # (e.g. employer) would otherwise be silently activated and served as
    # three equally-trusted facts. Rather than a new calibrated threshold
    # (which the reclassification was explicitly designed to avoid) or
    # reintroducing the quarantine this mechanism replaces, this follows
    # the project's existing honest-degradation convention (contested
    # facts, "unconfirmed"/"stale" freshness, migration 0018's own docstring):
    # never hidden, always marked. Surfaced in the served packet
    # (auto_reclassified) and the SDK-rendered prompt text so the reader --
    # human or agent -- can judge whether 3 "occurrences" actually look like
    # updates to one attribute instead.
    reclassified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Origin trust inherited from the source event (M8) — what authority
    # this fact was born with. Drives the consolidator's supersession
    # authority rule and the packet's provenance display.
    origin_trust: Mapped[str] = mapped_column(
        String(16), default="trusted", server_default="trusted"
    )

    # Temporal grounding (mechanism F1, migration 0020): {"start": iso,
    # "end": iso} when the source text described this fact with a RELATIVE
    # time expression ("last week", "il y a trois jours") that the
    # extractor resolved into a range anchored on the source event's
    # occurred_at -- see app.providers.base.ExtractedFact.temporal_range.
    # None for a fact carrying no such expression (an absolute date already
    # in `value`, or no time reference at all) -- deliberately NOT the same
    # as valid_from, which is always the MESSAGE's timestamp, not the
    # DESCRIBED event's.
    temporal_range: Mapped[dict | None] = mapped_column(JSONB)

    # When the fact is ABOUT (21 Aug), as opposed to when it became true
    # (`valid_from`, the message's own timestamp) or when Haki learned it
    # (`recorded_from`). "I got pre-approved back in August", said on 30
    # November: valid_from and recorded_from are November, observed_at is
    # August, and the answer to "when did you get pre-approved?" is
    # August.
    #
    # Derived at write time from `temporal_range.start` or from a single
    # unambiguous ISO date inside `value` -- see
    # app.consolidator.temporal. NULL for the many facts that are about no
    # particular instant ("I have a dog"), which is a real answer rather
    # than a failure: inventing one for them would make the column
    # meaningless for the facts that do have one.
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Provenance, down to the turn (21 Aug). The extraction prompt has
    # REQUIRED a verbatim quote from the source event since the write gate
    # (M1) landed -- a candidate that cannot produce one must be rejected
    # with reason "no_evidence_span" rather than written. Until now the
    # consolidator asked for it, validated it, and threw it away: the gate
    # demanded the proof and kept none of it.
    #
    # Persisting it buys three things at once: provenance a caller can
    # actually verify, the exact fact-to-turn link `source_chunk_id` below
    # needs, and a way to measure extraction quality per fact instead of
    # only in aggregate.
    evidence_span: Mapped[str | None] = mapped_column(String)
    # The episode chunk this fact was extracted from, resolved from
    # `evidence_span` at write time (app.consolidator._resolve_source_
    # chunk). NULL when the extractor produced no span, or when no chunk of
    # the source event contains it -- exact or absent, never approximate:
    # attributing a fact to the wrong turn would pollute that turn's index
    # (see app.consolidator._merge_facts_into_chunk_index), which is worse
    # than attributing nothing.
    #
    # ON DELETE SET NULL, not CASCADE: chunks are derived data that can be
    # rebuilt (scripts/backfill_episode_chunks.py), and losing a
    # rebuildable pointer must never take a ledger fact with it.
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("episode_chunks.id", ondelete="SET NULL")
    )

    # Retrieval (sprint 2): dense embedding + pre-rendered full-text column.
    # vector(1024) since migration 0029 (default embedder: local fastembed,
    # intfloat/multilingual-e5-large). Which model produced the stored
    # vectors is recorded once for the whole corpus in `embedding_space`
    # (migration 0028), not per row: it is a property of the corpus.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    search_text: Mapped[str | None] = mapped_column(String)
    # Precomputed tsvector of search_text (generated column, migration 0004):
    # ts_rank_cd reads it directly instead of re-parsing text on every query.
    # 'english', not 'simple' (migration 0023): 'simple' has no stopword
    # list, so a plain AND-tsquery over a natural-language question
    # requires "what"/"is"/"the" to literally appear in search_text (a
    # compact "predicate value" string that essentially never contains
    # them). Interpolated from settings.fts_config rather than hardcoded:
    # the text search configuration is a SCHEMA decision, frozen into the
    # generated column by whichever migration built it, and this
    # declaration must not drift away from that -- app.db.verify_fts_config
    # enforces the match at startup.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            f"to_tsvector('{settings.fts_config}', coalesce(search_text, ''))",
            persisted=True,
        ),
    )
