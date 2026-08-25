import uuid
from datetime import datetime
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
    # 3000 (was 2000, itself 900 before Sprint 2). The number changed
    # because what it MEASURES changed on 22 aout, not the packet: the
    # budget used to be charged against a stripped string while the prompt
    # carried the rendered line, so `2000` was really 4 565 tokens. It now
    # charges the line the caller receives, and 3000 of those buys the SAME
    # evidence the old 2000 did -- measured on eval.retrieval_bench, LoCoMo
    # 1-2, n=231, gold served against the REAL prompt cost:
    #
    #     before   4 565 real tokens -> 88.3 %
    #     after    3 223 real tokens -> 88.3 %      (-29 % for the same result)
    #     after    2 232 real tokens -> 84.4 %      (before, at 2 142: 80.5 %)
    #     after    4 219 real tokens -> 88.7 %
    #
    # It also puts the default back INSIDE the band the published curves
    # support, which the old one only appeared to be in: Zep/LoCoMo, LazyMem
    # and EMem agree the gain flattens well before 4 000 tokens with a
    # gpt-4o-mini-class reader (Zep/LoCoMo: +10.4pp from 347->1997 tok, then
    # +0.26 for the rest; LazyMem: top-50 WORSE than top-20). Haki was at
    # 4 565 and comparing itself against those curves as if it were at 2 000.
    #
    # The fixed instruction paragraphs (~290 tokens) are NOT taken out of
    # this: a caller cannot make them smaller. They are reported as
    # packet.overhead_tokens instead.
    budget_tokens: int = Field(default=3000)
    # 14 aout, mecanisme D: what "now" means for this call's freshness/
    # recency computations (volatility horizons, valid_to filter, recency
    # scoring term) -- defaults to the real wall clock. For replaying a
    # conversation dated in the past (an eval harness, a backfill), pass the
    # point in time the conversation's own timeline is at; a real caller in
    # normal operation should simply omit this. See app.context.build_context.
    as_of: datetime | None = None

    # Items the caller already holds from an earlier packet for this same
    # query (23 aout). Pass the `id` of a fact or the `episode_id` of an
    # episode; they are excluded before ranking, so the top-K slots go to
    # rows the caller does not already have.
    #
    # This is a NEXT PAGE, not a second hop, and the difference is
    # measured. On the questions whose first packet holds part of their
    # evidence, re-asking the SAME question with the seen items excluded
    # finds the missing turn 44.8 % of the time; re-asking with a query
    # reformulated from what the first packet said finds it 41.4 %, and
    # with that content alone 27.6 %. So: ask again with the SAME query.
    # Reformulating is measurably worse.
    #
    # What it buys is an adaptive budget -- the median call stays at
    # `budget_tokens`, and only the callers who find the answer missing pay
    # for a second page.
    exclude_ids: list[str] | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def exactly_one_subject(self) -> "ContextRequest":
        if (self.subject_id is None) == (self.subject_alias is None):
            raise ValueError("exactly one of subject_id or subject_alias is required")
        return self


class PacketFact(BaseModel):
    id: str
    # What the rendered block cites instead of a uuid -- see PacketRef,
    # defined below next to the episode that shares the scheme.
    ref: str | None = None
    # The exact line to print for this item -- see PacketLine.
    line: str | None = None
    predicate: str
    value: dict[str, Any]
    confidence: float | None
    valid_from: str | None
    # Trimmed of seconds and UTC offset -- see PacketEpisode.occurred_at_short.
    valid_from_short: str | None = None
    # Dual-date rendering (mechanism F1, 15 aout): exact offset from the
    # temporal point of view ("N days before/after the question"),
    # precomputed server-side so the reader verifies instead of
    # calculating. None only when valid_from itself is None. Defaults keep
    # old persisted traces (predating this field) re-validating unchanged.
    valid_from_relative: str | None = None
    # {"start": iso, "end": iso} when this fact's source text used a
    # relative time expression the extractor resolved -- see
    # app.providers.base.ExtractedFact.temporal_range. None otherwise.
    temporal_range: dict[str, str] | None = None
    # When the fact is ABOUT, normalised to one instant (migration 0029) --
    # distinct from valid_from, which is when it was SAID. Derived from
    # temporal_range.start or from a single unambiguous ISO date inside
    # `value`; None for the many facts about no particular instant.
    observed_at: str | None = None
    observed_at_relative: str | None = None
    # Reclassification safety net (16 aout): True when this fact was
    # activated by the automatic overflow reclassification (mechanism C)
    # rather than an extractor declaring memory_form="event" up front --
    # see Fact.reclassified_at. Default False keeps old persisted traces
    # (predating this field) re-validating unchanged.
    auto_reclassified: bool = False
    source_event_ids: list[str]
    # Typology + volatility (M2). All optional with None defaults: traces
    # persisted BEFORE migration 0016 re-validate through this model on
    # GET /v1/inspect/{trace_id} — an old packet without these keys must
    # keep loading (backward-compat guarantee, tested).
    fact_kind: str | None = None
    volatility: str | None = None
    last_confirmed: str | None = None
    freshness: str | None = None  # "current" | "unconfirmed" | "stale"
    # Provenance contract (M8): what authority this fact was born with, and
    # — for third_party origins — who actually said it. Defaults keep old
    # persisted traces (context_traces.packet) re-validating unchanged.
    origin_trust: str = "trusted"
    attributed_to: str | None = None
    # Open conflicts (13 aout): true when this fact is served alongside a
    # genuinely conflicting sibling instead of being hidden — see
    # app.context.CONTESTED_CONFLICT_MIN_MEMBERS. `conflict_id` correlates
    # both sides of the same disagreement. Defaults keep old persisted
    # traces (predating this field) re-validating unchanged.
    contested: bool = False
    conflict_id: str | None = None


# A packet-local reference (`F3`, `E7`), assigned in packing order and
# stable for the life of the packet (22 aout). It replaces the uuid the
# rendered block used to carry: a uuid4 is 35 o200k tokens against 2, and
# at ~46 identifiers per packet that was 23 % of everything sent to the
# model, spent on strings it is asked to cite and cannot carry reliably.
# The real ids stay right here in the packet, so a caller resolves a ref
# exactly -- see haki.runtime.resolve_refs in the SDK.
PacketRef = str

# The rendered line, server-side (22 aout). Until now the SAME block was
# built independently in the Python SDK, the TypeScript SDK and -- until
# P14 -- a third, poorer copy inside the MCP server, while the token budget
# was computed from a fourth string that matched none of them. Every marker
# added to one of them is a silent divergence in the others and in the
# budget.
#
# Rendering it once, where the packet is built, makes the budget exact by
# construction and leaves the SDKs to join lines and add the static header.
# Additive: an SDK that does not know this field renders from the structured
# fields exactly as before, and tests/test_packet_cost.py pins the two to
# the same string so the fallback cannot rot.
PacketLine = str


class PacketEpisode(BaseModel):
    """Source event excerpt served in the packet (episodic memory, sprint
    10): what happened, with its date and provenance id."""

    # The parent event: stable, addressable through /v1/timeline, and what
    # this field has always meant. Unchanged by the move to chunked
    # episodes (21 aout, migration 0027).
    event_id: str
    # What the rendered block cites instead of a uuid -- see PacketRef.
    ref: PacketRef | None = None
    # The exact line to print for this item -- see PacketLine.
    line: PacketLine | None = None
    # The ranked unit -- one turn-sized chunk of that event. Matches the
    # `episode_id` of the corresponding decision in the trace, so a served
    # episode can be correlated with the reason it was served. Additive:
    # every existing consumer reads `event_id` and is unaffected.
    episode_id: str | None = None
    kind: str
    occurred_at: str | None
    # The same instant without the seconds and the UTC offset, which answer
    # no question a packet is asked and cost 6 % of it (22 aout). An SDK
    # that does not know this field falls back to `occurred_at`.
    occurred_at_short: str | None = None
    # Dual-date rendering (mechanism F1, 15 aout) -- see
    # PacketFact.valid_from_relative. None only when occurred_at is None.
    occurred_at_relative: str | None = None
    excerpt: str
    # Context window (mechanism F2, 15 aout): True when this episode was
    # added as the temporal neighbor of an episode packed by score, or as
    # the source turn of a packed fact -- never a scored/ranked inclusion
    # of its own. False for an ordinary, score-packed episode.
    context_neighbor: bool = False


class ContextPacket(BaseModel):
    # What the rendered block costs BESIDES the items: the fixed
    # instruction paragraphs and the delimiters (22 aout). Not taken out of
    # budget_tokens -- a caller cannot make them smaller, and charging them
    # would turn every small budget into a silently empty packet -- but
    # reported, because `token_count + overhead_tokens` is what the prompt
    # actually costs and that number used to be invisible. 0 for an empty
    # packet, which renders as "".
    overhead_tokens: int = 0
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
