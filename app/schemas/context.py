import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.subjects import SubjectAliasIn

# Explicit noisy-failure contract (extends the gateway's X-Haki-Memory
# header — app/gateway/__init__.py — to every surface that returns a
# ContextPacket): "ok" = nothing to report, "degraded" = a packet was
# produced but something is worth flagging (open conflicts, a missing
# purpose, a caller-side degradation folded in via extra_warnings),
# "failed" = no real packet could be built at all (see
# app.context.failed_packet, used by callers that catch a build_context
# exception and still need to return a typed, inspectable result instead
# of silently swallowing it or blowing up the whole call).
ContextStatus = Literal["ok", "degraded", "failed"]


class ContextRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    # Exactly one of subject_id / subject_alias (M4 identity resolution).
    subject_id: str | None = Field(default=None, min_length=1, max_length=128)
    subject_alias: SubjectAliasIn | None = None
    query: str = Field(min_length=1)
    purpose: str | None = Field(default=None, max_length=128)
    budget_tokens: int = Field(default=900)

    @model_validator(mode="after")
    def exactly_one_subject(self) -> "ContextRequest":
        if (self.subject_id is None) == (self.subject_alias is None):
            raise ValueError("exactly one of subject_id or subject_alias is required")
        return self


class PacketFact(BaseModel):
    id: str
    predicate: str
    value: dict[str, Any]
    confidence: float | None
    valid_from: str | None
    source_event_ids: list[str]
    # Typology + volatility (M2). All optional with None defaults: traces
    # persisted BEFORE migration 0013 re-validate through this model on
    # GET /v1/inspect/{trace_id} — an old packet without these keys must
    # keep loading (backward-compat guarantee, tested).
    fact_kind: str | None = None
    volatility: str | None = None
    last_confirmed: str | None = None
    freshness: str | None = None  # "current" | "unconfirmed"


class PacketEpisode(BaseModel):
    """Source event excerpt served in the packet (episodic memory, sprint
    10): what happened, with its date and provenance id."""

    event_id: str
    kind: str
    occurred_at: str | None
    excerpt: str


class ContextPacket(BaseModel):
    facts: list[PacketFact]
    episodes: list[PacketEpisode] = Field(default_factory=list)
    # `warnings` doubles as the typed list of reasons for `status` — reused
    # rather than duplicated, since every warning is already a reason a
    # packet is not plainly "ok" (see build_context).
    warnings: list[str]
    status: ContextStatus = "ok"
    # M3 recall gate: "no_relevant_memory" when the relevance floor emptied
    # the packet although the subject HAS memories — deliberately NOT a
    # warning (a warning forces status="degraded"; this is an honest "ok").
    # None when disabled, when something was packed, or when the subject
    # truly has nothing.
    empty_reason: Literal["no_relevant_memory"] | None = None


class ContextResponse(BaseModel):
    packet: ContextPacket
    token_count: int
    trace_id: uuid.UUID


class TraceDecision(BaseModel):
    fact_id: str | None = None
    episode_id: str | None = None
    action: str  # included | excluded | blocked
    reason_code: str


class TraceResponse(BaseModel):
    trace_id: uuid.UUID
    project_id: str
    subject_id: str
    query: str
    purpose: str | None
    packet: ContextPacket
    decisions: list[TraceDecision]
    token_count: int
    duration_ms: int | None = None
    stage_timings: dict[str, int] | None = None
    fact_count: int | None = None
