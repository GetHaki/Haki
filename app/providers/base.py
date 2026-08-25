"""Interchangeable provider interfaces (PRD — "Providers IA").

Haki depends on these protocols only, never on a vendor SDK. Extraction and
embeddings are selected independently by configuration:

- extractor (`HAKI_LLM_PROVIDER=fake|openai`): LLM extraction of memory
  candidates from source events. Runs in the consolidator (async, off the
  hot path), so a remote LLM call is acceptable there.
- embedder (`HAKI_EMBED_PROVIDER=local|fake|openai`): text embeddings for
  facts AND context queries. The default is LOCAL (fastembed, ONNX CPU) so
  no network call ever sits in the `POST /v1/context` hot path.

Providers return raw candidates (`ExtractedFact` instances or plain dicts).
The consolidator ALWAYS re-validates every candidate with Pydantic before
touching the ledger, so a provider can never crash a batch with bad output.
"""

import logging
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models import FACT_KINDS, MEMORY_FORMS, VOLATILITY_CLASSES, Event

logger = logging.getLogger("haki.providers.base")

# Embedding dimension of the `embedding` columns (vector(1024), migration
# 0029). Every embedder selected by config MUST produce vectors of this size.
#
# This is what the CODE believes the schema is, which is not the same thing
# as what the schema IS -- an install that has not run `alembic upgrade
# head` disagrees with it, and the check in get_embedder cannot see that.
# The authority is the database: app.db.verify_embedding_space reads the
# real column width (and which model produced the stored vectors) at
# startup and refuses to serve on a mismatch. This constant exists so the
# fake embedder can produce vectors of the right width in tests, and as the
# cheap first check.
EMBEDDING_DIM = 1024

# Write gate (M1 — "porte d'ecriture"): reason codes for a candidate emitted
# with action="reject". A rejected candidate is counted/logged and NEVER
# becomes a Fact — see the taxonomy with worked examples in the extraction
# prompt (app/providers/openai.py _SYSTEM_PROMPT) and the anti-echo rule in
# app/consolidator (_echo_reject_reason), which assigns "echo_of_context"
# automatically regardless of what the provider itself returned.
#
# "imperative_directive" (added after M1): a candidate that is itself an
# instruction addressed to the agent/system — trying to steer FUTURE agent
# behavior ("ignore previous instructions", "always treat X as trustworthy",
# "never forget to always do Z") — rather than a fact about the subject. The
# risk this closes: a stored Fact is replayed verbatim into a future context
# packet, so a directive smuggled in as a "fact" would re-inject itself into
# the agent's instructions on every later turn. Narrow by design: Haki does
# not yet ingest untrusted third-party documents, only the agent's own
# conversation events, so this is one targeted rule, not a general prompt-
# injection defense (see app.consolidator module docstring and
# _imperative_directive_reason for the deterministic post-validation net and
# its documented residual false-positive/false-negative risk).
#
# "untrusted_instruction" (M8 — provenance as authority): a candidate that
# is a durable instruction (fact_kind="instruction") born from an event
# whose origin_trust is below semi_trusted (untrusted ingested content, or
# a third party in the subject's conversation). Complements
# imperative_directive: that rule catches orders aimed AT the agent
# whatever the origin; this one catches LEGITIMATE-looking durable
# instructions whose ORIGIN has no authority to steer future behavior —
# the write-time blind spot compositional/dormant attacks exploit. The
# provider may self-assign it (the prompt shows it the event's
# origin_trust), but the consolidator enforces it deterministically
# regardless (see app.consolidator._untrusted_instruction_reason).
REJECT_REASONS: tuple[str, ...] = (
    "echo_of_context",
    "system_noise",
    "config_dump",
    "transient_state",
    "unsupported_inference",
    "agent_self_reference",
    "no_evidence_span",
    "imperative_directive",
    "untrusted_instruction",
)


class ExtractedFact(BaseModel):
    """One memory candidate produced by an extraction provider."""

    # Chain-of-thought BEFORE the decision, not a summary after it (pattern
    # verified against Graphiti/Zep, GitHub issue #1666: putting a reasoning
    # field ahead of the verdict in a structured-output schema raised a
    # small model's contradiction-detection success from 47% to 93% — the
    # prompt requires providers to emit this key first so it conditions
    # predicate/action, not just documents them after the fact). Optional
    # here (FakeProvider and other providers never set it) — informational
    # only, never read downstream; app.consolidator addresses candidate
    # fields by name and never persists it onto a Fact row.
    reasoning: str | None = Field(default=None, max_length=2000)
    subject_id: str = Field(min_length=1, max_length=128)
    predicate: str = Field(min_length=1, max_length=128)
    value: dict[str, Any]
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    action: str = Field(default="create", pattern="^(create|supersede|reject)$")
    # Predicate of the active fact this candidate replaces (action=supersede).
    # Defaults to `predicate` when omitted.
    supersedes_predicate: str | None = Field(default=None, max_length=128)
    # Verbatim quote from the source event that grounds this candidate
    # (write gate M1). Optional in the schema — a provider that omits it
    # still validates — but the prompt REQUIRES it for action create/
    # supersede: a candidate the extractor cannot ground in an exact source
    # quote must be emitted as action="reject", reject_reason=
    # "no_evidence_span" instead of a bare, unsourced create/supersede.
    evidence_span: str | None = Field(default=None, max_length=4000)
    # Set when action="reject": why this candidate was screened out before
    # ever reaching the ledger. Required (validated below) when action is
    # "reject"; meaningless/ignored otherwise. One of REJECT_REASONS.
    reject_reason: str | None = Field(
        default=None, pattern="^(" + "|".join(REJECT_REASONS) + ")$"
    )
    # Typology + volatility (M2): proposed by the extractor, validated here,
    # overridable downstream. None = server defaults ("attribute"/"stable") --
    # a provider that never heard of these fields keeps working unchanged.
    fact_kind: str | None = Field(
        default=None, pattern="^(" + "|".join(FACT_KINDS) + ")$"
    )
    volatility: str | None = Field(
        default=None, pattern="^(" + "|".join(VOLATILITY_CLASSES) + ")$"
    )
    # Memory form (mechanism C, 15 aout): "state" (a scalar attribute that
    # changes over time -- the default, unchanged behavior) or "event" (an
    # accumulating occurrence -- volunteered somewhere, tried a restaurant,
    # attended an event -- where a new mention is a NEW fact, never a
    # replacement or a contradiction of the previous ones). None = server
    # default ("state" for a brand new identity; inherited from the
    # matched existing fact otherwise, see app.consolidator._apply_
    # candidate) -- a provider that never heard of this field keeps
    # working unchanged, exactly like fact_kind/volatility above.
    memory_form: str | None = Field(
        default=None, pattern="^(" + "|".join(MEMORY_FORMS) + ")$"
    )
    # Temporal grounding (mechanism F1, 15 aout): when the source text uses
    # a RELATIVE time expression for what this fact describes ("last week",
    # "il y a trois jours", "hier soir") -- as opposed to `valid_from`,
    # which is always the event's own `occurred_at` (when the message was
    # sent) -- this resolves that expression into an ISO range anchored on
    # `occurred_at`, {"start": ..., "end": ...}. Both keys required together
    # (a single invented point in time is worse than an honest range: "last
    # week" said on a Wednesday is a 7-day window, never a fabricated exact
    # day). None when the source text already carries an absolute date, or
    # states no time reference at all -- never guessed. The Chronos pattern
    # (current LongMemEval SOTA, 95.6%, precisely because it resolves this
    # at write time instead of leaving it for the reader to re-derive).
    temporal_range: dict[str, str] | None = Field(default=None)

    # Real-model bug found by external GTM-driven measurement (17 aout): a
    # live gpt-4o-mini extraction run on real LoCoMo conversations emitted
    # fact_kind="event" repeatedly -- not one of fact_kind's own three
    # values, but memory_form's. The two fields are documented separately
    # and clearly, but nothing stops a real (non-deterministic) model from
    # confusing "what kind of fact is this" with "is this an accumulating
    # occurrence" when both prompt sections use the word "event". Before
    # this fix, the strict `pattern=` constraint on fact_kind/volatility/
    # memory_form turned that single-field slip into total data loss --
    # the whole candidate (predicate, value, evidence_span, everything) was
    # rejected, not just the misclassified field. Every one of these three
    # fields already has a documented, safe default for "the provider never
    # set it" -- extending that same default to "the provider set it to
    # something outside the enum" recovers the fact instead of discarding
    # it, at the cost of a possibly-wrong classification tag rather than a
    # lost memory. Logged (not silent) so a real prompt-quality regression
    # is still visible, just no longer destructive.
    @field_validator("fact_kind", "volatility", "memory_form", mode="before")
    @classmethod
    def _coerce_unknown_enum_to_default(cls, value: Any, info: Any) -> Any:
        if value is None:
            return value
        allowed = {
            "fact_kind": FACT_KINDS,
            "volatility": VOLATILITY_CLASSES,
            "memory_form": MEMORY_FORMS,
        }[info.field_name]
        if value not in allowed:
            logger.warning(
                "extraction: provider set %s=%r, not one of %s -- falling back to "
                "the field's own default instead of dropping the whole candidate",
                info.field_name,
                value,
                sorted(allowed),
            )
            return None
        return value

    @model_validator(mode="after")
    def _reject_action_requires_reason(self) -> "ExtractedFact":
        if self.action == "reject" and self.reject_reason is None:
            raise ValueError("action 'reject' requires a reject_reason")
        return self

    @model_validator(mode="after")
    def _temporal_range_requires_both_bounds(self) -> "ExtractedFact":
        if self.temporal_range is None:
            return self
        if set(self.temporal_range) != {"start", "end"}:
            raise ValueError("temporal_range requires exactly 'start' and 'end'")
        for key in ("start", "end"):
            try:
                datetime.fromisoformat(self.temporal_range[key])
            except ValueError as exc:
                raise ValueError(f"temporal_range.{key} must be ISO 8601") from exc
        return self


# A provider may yield validated facts or raw dicts; both are accepted and
# re-validated by the consolidator.
RawCandidate = ExtractedFact | dict[str, Any]


@runtime_checkable
class Extractor(Protocol):
    async def extract_facts(
        self,
        events: list[Event],
        existing: list[dict[str, Any]] | None = None,
    ) -> list[RawCandidate]:
        """Extract memory candidates from source events.

        `existing` carries the subject's currently ACTIVE facts
        ({"predicate", "value", "valid_from"}) so the provider can decide
        `supersede` vs `create` with full knowledge — a change of mind must
        replace the old fact, not pile up a contradiction.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as PASSAGES, into EMBEDDING_DIM-dimensional vectors."""
        ...

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as QUERIES (22 aout).

        Separate from `embed` because a retrieval-trained model is trained
        asymmetrically -- a different prefix on the question side and on the
        passage side -- and using the passage form for a question loses
        recall without raising anything (measured: 89.6 % against 87.4 % on
        the retrieval bench with snowflake-arctic-embed-s). For a symmetric
        model this is the same function; implementing it is still required,
        so that "which side is this text on?" is answered by the embedder
        rather than assumed by every caller.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document's relevance to `query` with a cross-encoder
        (query+document jointly attended, not two separate embeddings
        compared by distance) -- the mechanism the literature ties to the
        single largest measured retrieval gain (SmartSearch ablation,
        +15.1pp, median gold rank 195 -> 8; see app.context's
        RERANK_TOP_K/HAKI_RERANK_ENABLED for where this is used).

        Returns scores aligned 1:1 with `documents`, same order, higher =
        more relevant. Not necessarily 0..1 or comparable across different
        reranker models/queries -- callers use these scores only to
        re-order candidates against EACH OTHER for the SAME query, never
        blended arithmetically with a different scoring scale.
        """
        ...
