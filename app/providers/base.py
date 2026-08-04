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

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from app.models import Event

# Embedding dimension of the `facts.embedding` column (vector(384), migration
# 0003). Every embedder selected by config MUST produce vectors of this size.
EMBEDDING_DIM = 384

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
REJECT_REASONS: tuple[str, ...] = (
    "echo_of_context",
    "system_noise",
    "config_dump",
    "transient_state",
    "unsupported_inference",
    "agent_self_reference",
    "no_evidence_span",
    "imperative_directive",
)


class ExtractedFact(BaseModel):
    """One memory candidate produced by an extraction provider."""

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

    @model_validator(mode="after")
    def _reject_action_requires_reason(self) -> "ExtractedFact":
        if self.action == "reject" and self.reject_reason is None:
            raise ValueError("action 'reject' requires a reject_reason")
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
        """Embed texts into EMBEDDING_DIM-dimensional vectors."""
        ...
