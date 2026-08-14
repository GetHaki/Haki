"""Context Assembler (PRD semaines 3-4).

Hybrid retrieval over the facts of one exact scope, then assembly under a
token budget, with a persisted decision trace.

Hard filters BEFORE scoring: status = active, exact (project_id, subject_id)
scope, valid_to IS NULL OR valid_to > now().

Open conflicts (13 aout, "stop hiding real conflicts"): a genuine two-sided
disagreement (an open ConflictSet with 2 members — the cap, see
CONFLICT_SET_MAX_MEMBERS) is now SERVED, both facts together, each marked
`contested`/`conflict_id` in the packet, rather than hidden. This relies on
the temporal tie-break fix (Bug 3, same day): once the answer prompt and
`build_prompt_context` both reliably resolve two dated conflicting values by
picking the most recent, showing both is strictly more informative than an
empty packet — the oracle@900 test that justified hiding them (3/3 failures
picking between two dated values with no working tie-break) no longer
applies. A single-member open set — a held/quarantined candidate (M8
untrusted origin, or the 3rd+ value once a pair is already capped, see
CONFLICT_SET_MAX_MEMBERS) — is NOT a disagreement to show, it is a
not-yet-trusted or not-yet-a-real-conflict value: still blocked outright,
reason_code conflict_open, exactly as before.
Post-retrieval, a volatility check (M2) degrades a fact past its freshness
horizon rather than hiding it: a slow fact is served flagged "unconfirmed",
a volatile/ephemeral fact is served flagged "stale" (14 aout — see
`_fact_freshness`; hard exclusion is reserved for superseded/deleted facts
and untrusted-origin instructions, never for "not recently reconfirmed").
`as_of` controls what "now" means for every freshness/recency computation
in one call — defaults to the real wall clock, unchanged for any caller
that omits it; see `build_context`.

Score = 0.6 * cosine similarity (pgvector embedding)
      + 0.25 * full-text rank (ts_rank_cd over the GENERATED search_vector
        column — computed once at write time, not re-parsed per query)
      + 0.15 * recency (exponential decay, 30-day time constant, on
        coalesce(valid_from, recorded_from)).

Multi-hop expansion (sprint 10): after the main pass, if budget remains,
a second full-text-only lookup seeded by entities found in the facts just
packed pulls in evidence that never matched the ORIGINAL query but becomes
relevant once the first hop is known (`reason_code=multi_hop_expansion`).
No LLM call, no extra embedding — bounded and one hop deep.

Latency (sprint 3) — two-phase retrieval, standard candidate-generation +
rerank: scoring ALL active facts of a scope costs ~200 ms at 10k facts, so
phase 1 generates candidates with the INDEXES (top RETRIEVAL_TOP_K by hnsw
cosine distance UNION top RETRIEVAL_TOP_K by GIN full-text rank), and phase
2 computes the full hybrid score on that small union only (≤ 2×TOP_K rows),
then caps at CANDIDATE_LIMIT. Only the columns needed for packing are
selected: decoding the 384-dim embedding of every returned row costs more
than the scoring itself (measured). Trade-off, documented: a fact that is
neither in the vector top-K nor in the full-text top-K cannot be served even
if recency would have lifted it — and facts beyond the cap are not traced.

Budget: tokens estimated as max(1, len(text) // 4). Facts and episodes are
packed from ONE ranked pool (key merging, 13 aout) — episodes scored on
the same weighted formula as facts, similarity + recency, minus the
full-text term facts get and episodes structurally can't yet (see
EPISODE_W_SIMILARITY/EPISODE_W_RECENCY) — rather than two separately-
budgeted pools merged by a fixed share (the interim fix shipped 12 aout,
after the same question set/budget answered from raw episode text instead
of extracted facts alone scored +46.6 points: a fact is compact and
durable, but lossy; the source wording it was extracted from is not). The
rest is excluded with reason_code over_budget. Every decision is written
to context_traces. Multi-hop expansion (below) stays a facts-only bolt-on
outside the unified pool — it was never on the same score scale as the
primary pass either, before or after this change.

Recall gate (M3): when settings.recall_max_distance > 0, candidates whose
cosine distance to the query exceeds it are excluded (reason_code
below_relevance_floor) before packing — facts and episodes alike. A call
where the gate empties the packet returns empty_reason="no_relevant_memory"
with status "ok": not a failure, an honest "nothing relevant enough".
"""

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Float, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app import metrics
from app.config import settings
from app.errors import ApiError
from app.models import ConflictSet, ContextTrace, Event, Fact, FactStatus, SubjectAlias
from app.providers import Embedder, get_embedder

# Scoring weights (documented; not part of the public contract).
W_SIMILARITY = 0.6
W_FULLTEXT = 0.25
W_RECENCY = 0.15
RECENCY_TAU_SECONDS = 30 * 86400  # exponential decay time constant: 30 days

# An open ConflictSet with this many members or more is a genuine two-sided
# disagreement, not a held/quarantined single candidate — see the "Open
# conflicts" paragraph above. Mirrors app.consolidator.CONFLICT_SET_MAX_
# MEMBERS (also 2, the cap); not imported from there directly, since
# app.consolidator already imports FROM this module (episode_text) and a
# reverse import would cycle.
CONTESTED_CONFLICT_MIN_MEMBERS = 2

# Max rows hydrated/packed per context call. With the default 900-token
# budget and facts as small as ~5 estimated tokens, ~180 facts can fit;
# 256 leaves headroom. Facts beyond the cap are not traced (see docstring).
CANDIDATE_LIMIT = 256

# Phase-1 candidate generation: top-K by vector distance (hnsw index) UNION
# top-K by full-text rank (GIN index); only this union gets the full hybrid
# score. 64 keeps recall comfortable while bounding the phase-2 work.
RETRIEVAL_TOP_K = 64

# M3 recall gate — floor on the SEMANTIC axis only (cosine distance of the
# candidate to the query), never on the hybrid score: similarity is the only
# bounded, embedder-calibratable term and carries the dominant weight (0.6);
# ts_rank_cd is unbounded and the "simple" ts config keeps stopwords, so any
# shared function word yields a nonzero rank (a lexical escape hatch would
# let noise through); recency says nothing about relevance. Distinct from
# SEMANTIC_MATCH_MAX_DISTANCE (0.28, fact<->fact paraphrases): query<->fact
# distances are structurally larger. Calibrated for the LOCAL embedder with
# scripts/check_recall_floor.py — re-run it before changing this value.
# Measured margin is narrower than the fact<->fact one: relevant queries
# topped out at 0.7046 ("combien de velos possede-t-il ?" vs bike_count),
# off-topic ones bottomed out at 0.7884 — 0.75 sits at the midpoint, but the
# gap (0.084) leaves far less slack than SEMANTIC_MATCH_MAX_DISTANCE's; a
# lexically-terse query on a short predicate is the tightest case measured.
# settings.recall_max_distance (HAKI_RECALL_MAX_DISTANCE) holds the ACTIVE
# threshold; 0.0 = gate disabled (default).
RECOMMENDED_RECALL_MAX_DISTANCE = 0.75


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Episodic memory (sprint 10): how much of an event's payload feeds the
# embedding, and how much is shown in the packet excerpt.
EPISODE_TEXT_CHARS = 4000
# Was 300 (~75 tokens) — a source excerpt that short is barely more than a
# label, not enough to answer anything requiring the actual wording. Real
# eval evidence (12-13 aout): the SAME 15-question LongMemEval sample,
# SAME 4000-token budget, answered from raw source-session text instead of
# extracted facts, scored 73.3% against 26.7% for facts alone (+46.6
# points) — matches the published finding (LongMemEval, ICLR 2025, S5.2)
# that facts-only retrieval loses information relative to raw text, and
# that concatenating both ("key merging") beats indexing them as two
# separate pools merged by rank — now what this module does, see the
# unified pool below). Raised to match EPISODE_TEXT_CHARS: the token
# budget check is what should decide how much of an episode fits, not a
# second, tighter, unconditional truncation applied before that check ever
# runs.
EPISODE_EXCERPT_CHARS = EPISODE_TEXT_CHARS
# Top-K source events retrieved per context call (cosine, hnsw).
EPISODE_TOP_K = 8

# Episode scoring (key merging, 13 aout): facts and episodes are packed
# from ONE ranked pool, not two separate budgets — the previous fixed
# EPISODE_MIN_BUDGET_SHARE floor (12 aout) was an honest stopgap, not the
# real fix; this is the real fix. Episodes are scored with the SAME
# weighted formula as facts (similarity + recency), just missing the
# full-text term: `events` has no search_vector column yet (facts do,
# migration 0004) — no full-text index to rank against, so that axis is
# left at zero rather than faked. Renormalized so the two SCORED axes
# (similarity, recency) keep facts' relative 4:1 emphasis and the score
# stays on the same 0-1 scale as a fact's — not summed directly against
# W_SIMILARITY/W_RECENCY, which would cap episodes at 0.75 and rank them
# systematically below facts regardless of actual relevance. A fact can
# still out-rank an equally-similar episode via its extra full-text axis
# (a fact matched lexically as well as semantically SHOULD win) — that
# asymmetry is honest, not a bug: facts are compact structured text,
# lexical match is cheap signal for them; episodes are prose, where
# semantic + temporal proximity carry more of the signal. Heuristic
# renormalization, not empirically calibrated — recalibrate once episodes
# have their own full-text axis (giving them the real 3-term formula) or
# once enough real eval data exists to tune the ratio directly.
EPISODE_W_SIMILARITY = W_SIMILARITY / (W_SIMILARITY + W_RECENCY)
EPISODE_W_RECENCY = W_RECENCY / (W_SIMILARITY + W_RECENCY)


def episode_text(kind: str, payload: dict | None) -> str:
    """Text embedded/quoted for an episode: kind + serialized payload,
    truncated. Deterministic — the consolidator embeds exactly this."""
    serialized = json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)
    return f"{kind} {serialized}"[:EPISODE_TEXT_CHARS]


def episode_excerpt(kind: str, payload: dict | None) -> str:
    """Human/agent-readable excerpt for the packet — as much of the source
    text as EPISODE_EXCERPT_CHARS allows; the token-budget packing loop is
    what actually decides how much of it gets served."""
    return episode_text(kind, payload)[:EPISODE_EXCERPT_CHARS]


def _render(predicate: str, value: dict[str, Any]) -> str:
    return f"{predicate} {json.dumps(value, sort_keys=True, ensure_ascii=False)}"


# Multi-hop expansion (sprint 10, bounded/deterministic): a second full-text
# pass seeded by entities mentioned in the facts already packed, discovering
# evidence that isn't semantically close to the ORIGINAL query but becomes
# relevant only once the first hop is known (two facts linked only by a
# shared name, not by wording similarity to the question). No LLM call, no
# extra embedding call — full-text only, reusing the existing GIN index.
# Bounded to a small number of entities, one hop deep, never recurses.
# Pattern: SmartSearch "index-free" (arXiv:2603.15599) — rule-based entity
# discovery + reseeded search beats an unbounded/LLM-based hop.
MULTI_HOP_MAX_ENTITIES = 2
MULTI_HOP_MAX_PER_ENTITY = 5

_ENTITY_TOKEN_RE = re.compile(r"[A-ZÀ-Ý][a-zà-ÿ]{2,}")
# Capitalized words that are common sentence-starters, not proper nouns —
# excluded so the heuristic doesn't seed expansion on noise.
_ENTITY_STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "here", "and", "with",
    "when", "where", "what", "which", "who",
    "le", "la", "les", "un", "une", "des", "ce", "cette", "ces", "et", "avec",
}


def _candidate_entities(texts: list[str], exclude: set[str], limit: int) -> list[str]:
    """Rule-based entity discovery (no LLM, no NER model): capitalized
    tokens across `texts`, ranked by frequency, excluding common
    sentence-initial words and anything already covered by `exclude`
    (typically the original query's own words)."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in _ENTITY_TOKEN_RE.findall(text):
            key = token.lower()
            if key in _ENTITY_STOPWORDS or key in exclude:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:limit]]


async def _expand_via_entities(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    seed_texts: list[str],
    query_words: set[str],
    exclude_ids: set[uuid.UUID],
    now: Any,
) -> list[Any]:
    """Second full-text pass seeded by entities found in `seed_texts`
    (already-packed facts). Returns extra Fact rows (id/predicate/value/
    confidence/valid_from/source_event_ids), ranked by ts_rank within each
    entity's own query, never revisiting `exclude_ids`. Pure full-text
    (GIN index) — no embedding call, so it stays cheap enough for the hot
    path."""
    entities = _candidate_entities(seed_texts, exclude=query_words, limit=MULTI_HOP_MAX_ENTITIES)
    found: list[Any] = []
    seen: set[uuid.UUID] = set(exclude_ids)
    for entity in entities:
        entity_query = func.websearch_to_tsquery("simple", entity)
        stmt = (
            select(
                Fact.id,
                Fact.predicate,
                Fact.value,
                Fact.confidence,
                Fact.valid_from,
                Fact.source_event_ids,
                Fact.recorded_from,
                Fact.last_reinforced_at,
                Fact.fact_kind,
                Fact.volatility,
                Fact.origin_trust,
                Fact.qualifiers,
            )
            .where(
                Fact.project_id == project_id,
                Fact.subject_id == subject_id,
                Fact.status == FactStatus.active,
                (Fact.valid_to.is_(None)) | (Fact.valid_to > now),
                Fact.search_vector.op("@@")(entity_query),
            )
            .order_by(func.ts_rank_cd(Fact.search_vector, entity_query).desc(), Fact.id)
            .limit(MULTI_HOP_MAX_PER_ENTITY)
        )
        for row in (await session.execute(stmt)).all():
            if row.id in seen:
                continue
            seen.add(row.id)
            found.append(row)
    return found


# Entity-aware fact scoring (13 aout, LoCoMo diagnostic): a conversation
# involving two named people (LoCoMo's structure — the tracked subject and
# whoever else appears in it) gets ingested under ONE shared subject; the
# extraction prompt already tags a fact about someone other than the
# tracked subject with an explicit "person" key in its value (see
# app.providers.openai's ATTRIBUTION rules), but until now nothing at
# retrieval time used that tag. Measured effect (LoCoMo single-hop, run
# gh-31698210575): 88% of single-hop failures, and in every sampled case
# the served packet was either missing the named person's facts entirely
# or dominated by facts about the OTHER named person in the conversation.
# Reuses _ENTITY_TOKEN_RE/_ENTITY_STOPWORDS (same rule-based, no-LLM
# detection already used for multi-hop expansion) rather than inventing a
# second entity-detection mechanism.
#
# Deliberately conservative: a fact with no "person" key — Haki's typical
# single-user product usage, where nearly every fact belongs to the
# tracked subject by construction — is NEVER touched. This only ever
# activates when BOTH sides identify a specific individual: the query
# names someone, AND the fact is explicitly tagged as being about someone
# (possibly a different someone). Magnitudes chosen to re-rank, not
# exclude: a mismatched-person fact can still win if nothing else is
# remotely relevant, same principle as the recall gate never forcing a
# hard zero.
ENTITY_MATCH_BOOST = 1.3
ENTITY_MISMATCH_PENALTY = 0.3


def _query_entities(query: str) -> set[str]:
    return {t for t in _ENTITY_TOKEN_RE.findall(query) if t.lower() not in _ENTITY_STOPWORDS}


def _entity_adjusted_score(score: float, value: Any, query_entities: set[str]) -> float:
    if not query_entities or not isinstance(value, dict):
        return score
    person = value.get("person")
    if not isinstance(person, str) or not person:
        return score
    return score * (ENTITY_MATCH_BOOST if person in query_entities else ENTITY_MISMATCH_PENALTY)


# Volatility (M2): freshness horizon per class, read from config at call
# time (env-overridable, never hardcoded). "stable" has no horizon — the
# pre-M2 behavior every existing fact keeps.
def volatility_horizon(volatility: str) -> timedelta | None:
    days = {
        "slow": settings.volatility_horizon_slow_days,
        "volatile": settings.volatility_horizon_volatile_days,
        "ephemeral": settings.volatility_horizon_ephemeral_days,
    }.get(volatility)
    return timedelta(days=days) if days is not None else None


def _fact_freshness(row: Any, now: datetime) -> str:
    """"current" | "unconfirmed" | "stale" for one retrieved fact row.

    Clock: coalesce(last_reinforced_at, valid_from, recorded_from) — a
    write-time reinforcement (migration 0015) already refreshes
    last_reinforced_at on every re-assertion of the same value, so it
    doubles as the freshness clock without a separate column, measured
    against `now` (real wall-clock time, or the caller's `as_of` — see
    `build_context`).

    14 aout (mecanisme D, research/Diagnostic_Couverture_2026-08-14.md): a
    volatile/ephemeral fact past its horizon used to be excluded outright
    ("expired", hard gate) -- changed to served, flagged "stale", same
    honest-degradation treatment a "slow" fact already got as "unconfirmed".
    Exclusion was never the right contract for "we have not reconfirmed this
    recently": the fact is not FALSE, it is UNCERTAIN, and hiding it left the
    agent knowing neither the value nor that a value existed. Measured
    effect on a benchmark harness that replays conversations dated years in
    the past against the real clock: ~38% of a subject's facts hidden on
    every single query regardless of relevance, since a 7-60 day horizon is
    dwarfed by a multi-year gap -- not a benchmark-only artifact, the same
    mechanism degrades any real subject who returns after a long pause.
    Hard exclusion (never served as current) stays reserved for facts that
    are actually gone: superseded, deleted, or an untrusted-origin
    instruction (M8) -- see the module docstring.
    """
    horizon = volatility_horizon(getattr(row, "volatility", "stable") or "stable")
    if horizon is None:
        return "current"
    reference = row.last_reinforced_at or row.valid_from or row.recorded_from
    if reference is None or now - reference <= horizon:
        return "current"
    return "unconfirmed" if row.volatility == "slow" else "stale"


def _packet_fact(
    row: Any, freshness: str = "current", *, conflict_id: str | None = None
) -> dict[str, Any]:
    reference = row.last_reinforced_at or row.valid_from or row.recorded_from
    return {
        "id": str(row.id),
        "predicate": row.predicate,
        "value": row.value,
        "confidence": row.confidence,
        "valid_from": row.valid_from.isoformat() if row.valid_from else None,
        "source_event_ids": [str(e) for e in row.source_event_ids],
        "fact_kind": row.fact_kind,
        "volatility": row.volatility,
        # Honest freshness contract (M2): when the fact was last confirmed
        # by an event, and whether it is inside its volatility horizon.
        "last_confirmed": reference.isoformat() if reference else None,
        "freshness": freshness,
        # Provenance contract (M8): what authority this fact was born with,
        # and who actually said it when a third party did.
        "origin_trust": row.origin_trust or "trusted",
        "attributed_to": (row.qualifiers or {}).get("attributed_to"),
        # Open conflicts (13 aout): set together with its sibling(s) from the
        # same open ConflictSet when this fact is one half of a genuine
        # two-sided disagreement being served rather than hidden — see
        # CONTESTED_CONFLICT_MIN_MEMBERS. None for an ordinary fact.
        "contested": conflict_id is not None,
        "conflict_id": conflict_id,
    }


async def _open_conflict_sets(
    session: AsyncSession, *, project_id: str, subject_id: str
) -> list[ConflictSet]:
    stmt = select(ConflictSet).where(
        ConflictSet.project_id == project_id,
        ConflictSet.subject_id == subject_id,
        ConflictSet.status == "open",
    )
    return list((await session.execute(stmt)).scalars().all())


def failed_packet(reasons: list[str]) -> dict[str, Any]:
    """Canonical empty packet, status="failed" (see ContextStatus).

    For a caller that CAUGHT a build_context exception (build_context
    itself never returns a half-built packet — it raises, same as any
    other exception) and still needs to hand back a typed, inspectable
    result instead of either swallowing the failure silently or letting it
    blow up the whole request. Mirrors the gateway's existing
    `X-Haki-Memory: degraded` header contract (app/gateway/__init__.py) so
    every surface that can serve a ContextPacket — MCP tools, a future SDK
    helper — reports failure the same, loud way. Never persisted as a
    context_traces row: the caller decides what (if anything) to do with
    it.
    """
    return {
        "facts": [],
        "episodes": [],
        "warnings": list(reasons),
        "status": "failed",
        "empty_reason": None,
    }


async def build_context(
    session: AsyncSession,
    *,
    project_id: str,
    subject_id: str,
    query: str,
    purpose: str | None = None,
    budget_tokens: int = 900,
    embedder: Embedder | None = None,
    extra_warnings: list[str] | None = None,
    as_of: datetime | None = None,
) -> tuple[dict[str, Any], int, uuid.UUID]:
    """Assemble a ContextPacket. Returns (packet, token_count, trace_id).

    `extra_warnings` (e.g. policy warnings computed by the caller) are
    appended to the packet warnings BEFORE the trace is persisted, so the
    inspection trace shows exactly what the API returned.

    `as_of` (14 aout, mecanisme D): the point in time "now" means for every
    freshness computation in this call -- volatility horizons, the
    valid_to scope filter, and the recency term of both scoring formulas.
    Defaults to the real wall clock (`func.now()`), unchanged behavior for
    every caller that omits it. Exists because Haki's own ledger is
    bitemporal (occurred_at vs recorded_at) but retrieval read it against
    the SERVER's clock regardless -- fine for a real subject whose
    conversations track real time, silently wrong for anything replayed
    from a fixed point in the past (an eval harness, a backfill, a subject
    resuming after a long gap): a volatile fact from "last week" relative
    to the conversation looked years-stale relative to whenever this
    function happens to run. See research/Diagnostic_Couverture_2026-08-14.md
    for the measured effect (~38% of facts hidden on every query on the
    LoCoMo/LongMemEval harnesses before this parameter existed).
    """
    if budget_tokens <= 0:
        raise ApiError(
            type="budget_exceeded",
            message="budget_tokens must be a positive integer",
            field="budget_tokens",
        )
    embedder = embedder or get_embedder()
    request_start = time.perf_counter()
    stage_timings: dict[str, int] = {}

    embed_start = time.perf_counter()
    query_embedding = (await embedder.embed([query]))[0]
    stage_timings["embed"] = round((time.perf_counter() - embed_start) * 1000)

    # A bound literal (not just a Python datetime) so it can take part in
    # SQL arithmetic (`now - Fact.valid_from`) exactly like func.now() does.
    now = literal(as_of) if as_of is not None else func.now()
    similarity = func.coalesce(1 - Fact.embedding.cosine_distance(query_embedding), 0.0)
    # websearch_to_tsquery accepts arbitrary user text (plain to_tsquery would
    # raise on queries without &/| operators). search_vector is a GENERATED
    # column: the tsvector is built once at write time (migration 0004).
    ts_query = func.websearch_to_tsquery("simple", query)
    fulltext = func.coalesce(func.ts_rank_cd(Fact.search_vector, ts_query), 0.0)
    recency = func.exp(
        -func.extract(
            "epoch", now - func.coalesce(Fact.valid_from, Fact.recorded_from)
        )
        / RECENCY_TAU_SECONDS
    )
    score = (
        W_SIMILARITY * similarity + W_FULLTEXT * fulltext + W_RECENCY * recency
    ).cast(Float)

    # Hard filters first: exact scope, active only, still valid.
    scope_filters = [
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.status == FactStatus.active,
        (Fact.valid_to.is_(None)) | (Fact.valid_to > now),
    ]

    # Phase 1 — candidate generation with the indexes (hnsw + GIN). Without
    # this, phase 2 would score every active fact of the scope (~200 ms at
    # 10k facts in the sprint-3 benchmark).
    #
    # Every ORDER BY below carries `Fact.id` as a secondary key (see the
    # same reasoning at the phase-2 query): ties on the primary key are a
    # real, common case here, not a theoretical one -- with no secondary
    # key, Postgres is free to return a tied group in whatever order its
    # query plan happens to produce, which can and does shift between
    # otherwise-identical calls (confirmed: scripts/
    # check_retrieval_discrimination.py, run twice against the same
    # project/subject/query/budget, returned two DIFFERENT fact sets).
    vector_top = (
        select(Fact.id)
        .where(*scope_filters)
        .order_by(Fact.embedding.cosine_distance(query_embedding), Fact.id)
        .limit(RETRIEVAL_TOP_K)
        .cte("vector_top")
    )
    fts_top = (
        select(Fact.id)
        .where(*scope_filters, Fact.search_vector.op("@@")(ts_query))
        .order_by(func.ts_rank_cd(Fact.search_vector, ts_query).desc(), Fact.id)
        .limit(RETRIEVAL_TOP_K)
        .cte("fts_top")
    )
    candidates = select(vector_top.c.id).union(select(fts_top.c.id)).cte("candidates")

    # Phase 2 — full hybrid score on the candidate union only. Only the
    # columns needed for packing are selected: decoding the 384-dim embedding
    # of every returned row costs more than the scoring itself (measured in
    # the sprint-3 benchmark). The cap keeps the work flat no matter how many
    # facts the scope holds.
    #
    # `Fact.id` as a secondary sort key (13 aout, "Bug 2" diagnostic, 11
    # aout): a fact's recency score depends only on coalesce(valid_from,
    # recorded_from) -- facts written in the same consolidation batch
    # routinely share that value down to the minute. For an off-topic
    # query (similarity and full-text both exactly 0), recency is the
    # ENTIRE score, so every fact in such a batch ties EXACTLY, not just
    # approximately. `score.desc()` alone leaves that tie's order to
    # Postgres's query plan, which is not guaranteed stable between two
    # otherwise-identical calls -- confirmed empirically (see
    # scripts/check_retrieval_discrimination.py): the same project,
    # subject, query and budget returned two different fact sets on
    # consecutive runs, purely from tie order, not from any real
    # relevance signal. This was the ACTUAL mechanism behind the
    # originally-reported "Bug 2" symptom (five different questions
    # returning an identical packet) once budget headroom alone was ruled
    # out on a high-volume subject (see the same script): a low-signal
    # query hits a wide tie, and without a stable tiebreaker, "which facts
    # win" is not reproducible even for the SAME query run twice.
    stmt = (
        select(
            Fact.id,
            Fact.predicate,
            Fact.value,
            Fact.confidence,
            Fact.valid_from,
            Fact.source_event_ids,
            Fact.recorded_from,
            Fact.last_reinforced_at,
            Fact.fact_kind,
            Fact.volatility,
            Fact.origin_trust,
            Fact.qualifiers,
            Fact.embedding.cosine_distance(query_embedding).label("distance"),
            score.label("score"),
        )
        .where(Fact.id.in_(select(candidates.c.id)))
        .order_by(score.desc(), Fact.id)
        .limit(CANDIDATE_LIMIT)
    )
    retrieval_start = time.perf_counter()
    rows = (await session.execute(stmt)).all()
    stage_timings["retrieval"] = round((time.perf_counter() - retrieval_start) * 1000)

    # Split open conflicts (13 aout): a genuine 2-member disagreement is
    # served, contested, sibling alongside sibling; a single-member set (a
    # held/quarantined candidate) stays fully blocked — see
    # CONTESTED_CONFLICT_MIN_MEMBERS and the module docstring.
    open_conflicts = await _open_conflict_sets(
        session, project_id=project_id, subject_id=subject_id
    )
    quarantined_ids: set[uuid.UUID] = set()
    contested_conflict_by_fact: dict[uuid.UUID, ConflictSet] = {}
    for conflict in open_conflicts:
        if len(conflict.fact_ids) >= CONTESTED_CONFLICT_MIN_MEMBERS:
            for fid in conflict.fact_ids:
                contested_conflict_by_fact[fid] = conflict
        else:
            quarantined_ids.update(conflict.fact_ids)

    # Every member of a contested set, fetched once regardless of status —
    # the losing side of a genuine disagreement stays `candidate` (never
    # scored by the phase-2 query above, which filters status == active)
    # and needs to be hydrated directly so it can be packed alongside its
    # active sibling.
    contested_rows_by_id: dict[uuid.UUID, Fact] = {}
    if contested_conflict_by_fact:
        contested_members = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.id.in_(contested_conflict_by_fact.keys())
                    )
                )
            )
            .scalars()
            .all()
        )
        contested_rows_by_id = {f.id: f for f in contested_members}

    now_dt = as_of or datetime.now(timezone.utc)
    decisions: list[dict[str, Any]] = []
    eligible: list[Any] = []
    freshness_by_id: dict[uuid.UUID, str] = {}
    for row in rows:
        if row.id in quarantined_ids:
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "blocked",
                    "reason_code": "conflict_open",
                }
            )
            continue
        # 14 aout (mecanisme D): "stale" (volatile/ephemeral past horizon) is
        # no longer excluded here -- it is served like "unconfirmed" always
        # was, annotated in the packet via freshness_by_id/_packet_fact.
        freshness_by_id[row.id] = _fact_freshness(row, now_dt)
        eligible.append(row)

    # A held/quarantined candidate is blocked too (it is not active, so it
    # never entered the scored `rows` above at all) — traced regardless of
    # whether it would otherwise have matched this query, same as before
    # 13 aout. Only single-member sets: a contested pair's candidate side
    # is handled separately, packed alongside its active sibling below.
    if quarantined_ids:
        quarantined_candidates = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.id.in_(quarantined_ids),
                        Fact.status == FactStatus.candidate,
                    )
                )
            )
            .scalars()
            .all()
        )
        for fact in quarantined_candidates:
            decisions.append(
                {
                    "fact_id": str(fact.id),
                    "action": "blocked",
                    "reason_code": "conflict_open",
                }
            )

    # Episodic memory (sprint 10, key merging 13 aout): the most relevant
    # SOURCE EVENTS of the same scope, scored on the SAME weighted formula
    # as facts — similarity + recency, see EPISODE_W_SIMILARITY/
    # EPISODE_W_RECENCY — so they compete fairly in ONE ranked pool below,
    # not two separately-budgeted ones. This is what answers "what
    # happened / when" questions, and (12-13 aout finding) carries
    # information a compact fact loses entirely: the extractor keeps
    # durable facts only, episodes keep the dated events in their own
    # words.
    episode_similarity = func.coalesce(
        1 - Event.embedding.cosine_distance(query_embedding), 0.0
    )
    episode_recency = func.exp(
        -func.extract("epoch", now - Event.occurred_at) / RECENCY_TAU_SECONDS
    )
    episode_score = (
        EPISODE_W_SIMILARITY * episode_similarity + EPISODE_W_RECENCY * episode_recency
    ).cast(Float)
    episodes_start = time.perf_counter()
    episode_rows = (
        (
            await session.execute(
                select(
                    Event.id,
                    Event.kind,
                    Event.occurred_at,
                    Event.payload,
                    Event.embedding.cosine_distance(query_embedding).label("distance"),
                    episode_score.label("score"),
                )
                .where(
                    Event.project_id == project_id,
                    Event.subject_id == subject_id,
                    Event.embedding.is_not(None),
                    # Provenance guard (M8): untrusted-origin events are
                    # never served as episodes — an episode is a VERBATIM
                    # payload excerpt replayed into the agent's context,
                    # i.e. a direct injection channel for ingested content.
                    # Their extracted facts already go through the
                    # quarantine path; the raw payload must not bypass it.
                    Event.origin_trust != "untrusted",
                )
                .order_by(episode_score.desc(), Event.id)
                .limit(EPISODE_TOP_K)
            )
        )
        .all()
    )
    stage_timings["episodes"] = round((time.perf_counter() - episodes_start) * 1000)

    # Unified ranked pool (key merging): facts and episodes, both filtered
    # by the SAME recall floor (M3, semantic/distance axis only — never the
    # composite score, see its module-level comment), merged into one list
    # by their comparable score, packed greedily against budget_tokens in a
    # SINGLE pass. A fact and an episode compete on their actual merits now,
    # not on which separately-budgeted pool they happened to land in.
    recall_max_distance = settings.recall_max_distance
    query_entities = _query_entities(query)
    pool: list[tuple[float, str, Any]] = []
    for row in eligible:
        if recall_max_distance > 0 and (
            row.distance is None or row.distance > recall_max_distance
        ):
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "excluded",
                    "reason_code": "below_relevance_floor",
                }
            )
            continue
        pool.append((_entity_adjusted_score(row.score, row.value, query_entities), "fact", row))
    for row in episode_rows:
        if recall_max_distance > 0 and (
            row.distance is None or row.distance > recall_max_distance
        ):
            decisions.append(
                {
                    "episode_id": str(row.id),
                    "action": "excluded",
                    "reason_code": "below_relevance_floor",
                }
            )
            continue
        pool.append((row.score, "episode", row))
    pool.sort(key=lambda item: item[0], reverse=True)

    packet_facts: list[dict[str, Any]] = []
    packet_episodes: list[dict[str, Any]] = []
    packed_fact_ids: set[uuid.UUID] = set()
    token_count = 0
    for _score, kind, row in pool:
        if kind == "fact":
            cost = estimate_tokens(_render(row.predicate, row.value))
        else:
            excerpt = episode_excerpt(row.kind, row.payload)
            cost = estimate_tokens(f"{row.occurred_at:%Y-%m-%d %H:%M} {row.kind} {excerpt}")
        if token_count + cost > budget_tokens:
            if kind == "fact":
                decisions.append(
                    {"fact_id": str(row.id), "action": "excluded", "reason_code": "over_budget"}
                )
            else:
                decisions.append(
                    {"episode_id": str(row.id), "action": "excluded", "reason_code": "over_budget"}
                )
            continue
        token_count += cost
        if kind == "fact":
            conflict = contested_conflict_by_fact.get(row.id)
            packet_facts.append(
                _packet_fact(
                    row,
                    freshness_by_id.get(row.id, "current"),
                    conflict_id=str(conflict.id) if conflict else None,
                )
            )
            packed_fact_ids.add(row.id)
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "included",
                    "reason_code": "conflict_disputed" if conflict else "top_score",
                }
            )
            if conflict is not None:
                # Serve the sibling(s) of this genuine disagreement right
                # alongside it (13 aout): the losing side never entered the
                # scored pool on its own (still `candidate`, filtered out of
                # phase 2 above) — without this, the pool would only ever
                # surface the winning/active half, defeating the point of
                # showing the conflict instead of hiding it.
                for sibling_id in conflict.fact_ids:
                    if sibling_id == row.id or sibling_id in packed_fact_ids:
                        continue
                    sibling = contested_rows_by_id.get(sibling_id)
                    if sibling is None:
                        continue
                    sibling_freshness = _fact_freshness(sibling, now_dt)
                    sibling_cost = estimate_tokens(_render(sibling.predicate, sibling.value))
                    if token_count + sibling_cost > budget_tokens:
                        decisions.append(
                            {
                                "fact_id": str(sibling_id),
                                "action": "excluded",
                                "reason_code": "over_budget",
                            }
                        )
                        continue
                    token_count += sibling_cost
                    packet_facts.append(
                        _packet_fact(sibling, sibling_freshness, conflict_id=str(conflict.id))
                    )
                    packed_fact_ids.add(sibling_id)
                    decisions.append(
                        {
                            "fact_id": str(sibling_id),
                            "action": "included",
                            "reason_code": "conflict_disputed",
                        }
                    )
        else:
            packet_episodes.append(
                {
                    "event_id": str(row.id),
                    "kind": row.kind,
                    "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
                    "excerpt": excerpt,
                }
            )
            decisions.append(
                {"episode_id": str(row.id), "action": "included", "reason_code": "top_score"}
            )

    # Multi-hop expansion: only worth trying if the main pass packed
    # something to seed entities from, and budget remains. Stays a
    # facts-only bolt-on OUTSIDE the unified pool above: ts_rank_cd within
    # a per-entity query was never on the same score scale as the primary
    # hybrid/episode score, before or after key merging — appended directly
    # against whatever budget the unified pass left, not re-merged into it.
    # Multi-hop rows are deliberately NOT gated by the recall floor: they
    # are seeded only by gate-passing facts, and their whole point is
    # lexical evidence that is semantically far from the ORIGINAL query.
    if packet_facts and token_count < budget_tokens:
        multi_hop_start = time.perf_counter()
        query_words = {t.lower() for t in _ENTITY_TOKEN_RE.findall(query)}
        seed_texts = [_render(f["predicate"], f["value"]) for f in packet_facts]
        included_fact_ids = {uuid.UUID(f["id"]) for f in packet_facts}
        extra_rows = await _expand_via_entities(
            session,
            project_id=project_id,
            subject_id=subject_id,
            seed_texts=seed_texts,
            query_words=query_words,
            exclude_ids=included_fact_ids
            | quarantined_ids
            | set(contested_conflict_by_fact.keys()),
            now=now,
        )
        stage_timings["multi_hop_expansion"] = round(
            (time.perf_counter() - multi_hop_start) * 1000
        )
        for row in extra_rows:
            if token_count >= budget_tokens:
                break
            freshness = _fact_freshness(row, now_dt)
            cost = estimate_tokens(_render(row.predicate, row.value))
            if token_count + cost <= budget_tokens:
                packet_facts.append(_packet_fact(row, freshness))
                token_count += cost
                decisions.append(
                    {
                        "fact_id": str(row.id),
                        "action": "included",
                        "reason_code": "multi_hop_expansion",
                    }
                )

    # Fragmentation detector (M4): a subject with ZERO memory that is
    # registered as an alias of another subject is NOT a cold start — the
    # memories exist, they live under the canonical id. Loud warning =>
    # status "degraded" (existing noisy-failure contract), never a silent
    # empty packet. One indexed lookup (ix_subject_aliases_lookup), and only
    # on empty results — the hot path pays nothing.
    fragmentation_alias = None
    if not packet_facts and not packet_episodes:
        fragmentation_alias = (
            (
                await session.execute(
                    select(SubjectAlias)
                    .where(
                        SubjectAlias.project_id == project_id,
                        SubjectAlias.alias_value == subject_id,
                        SubjectAlias.canonical_subject_id != subject_id,
                    )
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    warnings: list[str] = []
    if fragmentation_alias is not None:
        warnings.append(
            "identity_fragmentation: subject "
            f"'{subject_id}' is registered as a '{fragmentation_alias.alias_kind}' "
            f"alias of '{fragmentation_alias.canonical_subject_id}' — its memories "
            "live under the canonical subject; query the canonical id or pass "
            "subject_alias so the server resolves it"
        )
    n_blocked = sum(1 for d in decisions if d["reason_code"] == "conflict_open")
    if n_blocked:
        warnings.append(
            f"open_conflict: {n_blocked} fact(s) hidden pending conflict resolution"
        )
    n_disputed = sum(
        1
        for d in decisions
        if d["reason_code"] == "conflict_disputed" and d["action"] == "included"
    )
    if n_disputed:
        warnings.append(
            f"open_conflict: {n_disputed} fact(s) served with an unresolved "
            "conflicting value alongside them — apply the most recent "
            "'valid from' date to determine which is current"
        )
    # 14 aout (mecanisme D): volatility no longer hides anything -- past its
    # horizon a fact is served flagged "stale" (see _fact_freshness), same
    # silent-in-warnings treatment "unconfirmed" already got. No aggregate
    # warning here on purpose, for the same reason: it is per-fact honest
    # degradation, not a packet-level problem worth surfacing as a warning.
    warnings.extend(extra_warnings or [])

    # Noisy-failure contract (ContextStatus): "degraded" whenever there is
    # something worth flagging (open conflicts, a caller-supplied warning
    # such as the missing-purpose policy notice). build_context never
    # produces "failed" itself — an internal failure raises (loud, same as
    # any other exception in this function); "failed" is for a CALLER that
    # catches that exception (see `failed_packet`).
    status = "degraded" if warnings else "ok"
    metrics.increment(f"context.{status}")

    # M3: honest empty packet — the gate emptied the packet although
    # candidates existed. NOT a failure and NOT "this subject has no
    # memory": status stays "ok", warnings stay untouched (a warning would
    # force "degraded" — see above), the dedicated field carries the signal
    # and the below_relevance_floor decisions in the trace explain it.
    empty_reason = None
    if (
        recall_max_distance > 0
        and not packet_facts
        and not packet_episodes
        and any(d["reason_code"] == "below_relevance_floor" for d in decisions)
    ):
        empty_reason = "no_relevant_memory"
        metrics.increment("context.empty_no_relevant_memory")

    packet = {
        "facts": packet_facts,
        "episodes": packet_episodes,
        "warnings": warnings,
        "status": status,
        "empty_reason": empty_reason,
    }

    trace = ContextTrace(
        project_id=project_id,
        subject_id=subject_id,
        query=query,
        purpose=purpose,
        packet=packet,
        decisions=decisions,
        token_count=token_count,
        duration_ms=round((time.perf_counter() - request_start) * 1000),
        stage_timings=stage_timings,
        fact_count=len(packet_facts),
    )
    session.add(trace)
    await session.flush()
    return packet, token_count, trace.id


async def get_trace(
    session: AsyncSession,
    *,
    trace_id: uuid.UUID,
    project_id: str,
    subject_id: str,
) -> ContextTrace:
    """Return a trace, scoped: a trace never leaks outside its scope.

    A scope mismatch returns the same 404 as an unknown id, so the existence
    of a trace is never revealed across scopes.
    """
    trace = await session.get(ContextTrace, trace_id)
    if (
        trace is None
        or trace.project_id != project_id
        or trace.subject_id != subject_id
    ):
        raise ApiError(
            type="trace_not_found",
            message=f"Trace {trace_id} does not exist",
            field="trace_id",
            status_code=404,
        )
    return trace
