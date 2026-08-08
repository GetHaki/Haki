"""Context Assembler (PRD semaines 3-4).

Hybrid retrieval over the facts of one exact scope, then assembly under a
token budget, with a persisted decision trace.

Hard filters BEFORE scoring: status = active, exact (project_id, subject_id)
scope, valid_to IS NULL OR valid_to > now(). Facts listed in an OPEN conflict
set are never served: they are blocked with reason_code conflict_open.

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

Budget: tokens estimated as max(1, len(text) // 4); facts are packed by
decreasing score until the budget is full, the rest is excluded with
reason_code over_budget. Every decision is written to context_traces.
"""

import json
import re
import time
import uuid
from typing import Any

from sqlalchemy import Float, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app import metrics
from app.errors import ApiError
from app.models import ConflictSet, ContextTrace, Event, Fact, FactStatus
from app.providers import Embedder, get_embedder

# Scoring weights (documented; not part of the public contract).
W_SIMILARITY = 0.6
W_FULLTEXT = 0.25
W_RECENCY = 0.15
RECENCY_TAU_SECONDS = 30 * 86400  # exponential decay time constant: 30 days

# Max rows hydrated/packed per context call. With the default 900-token
# budget and facts as small as ~5 estimated tokens, ~180 facts can fit;
# 256 leaves headroom. Facts beyond the cap are not traced (see docstring).
CANDIDATE_LIMIT = 256

# Phase-1 candidate generation: top-K by vector distance (hnsw index) UNION
# top-K by full-text rank (GIN index); only this union gets the full hybrid
# score. 64 keeps recall comfortable while bounding the phase-2 work.
RETRIEVAL_TOP_K = 64


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Episodic memory (sprint 10): how much of an event's payload feeds the
# embedding, and how much is shown in the packet excerpt.
EPISODE_TEXT_CHARS = 4000
EPISODE_EXCERPT_CHARS = 300
# Top-K source events retrieved per context call (cosine, hnsw).
EPISODE_TOP_K = 8


def episode_text(kind: str, payload: dict | None) -> str:
    """Text embedded/quoted for an episode: kind + serialized payload,
    truncated. Deterministic — the consolidator embeds exactly this."""
    serialized = json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)
    return f"{kind} {serialized}"[:EPISODE_TEXT_CHARS]


def episode_excerpt(kind: str, payload: dict | None) -> str:
    """Short human/agent-readable excerpt for the packet (~300 chars)."""
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
            )
            .where(
                Fact.project_id == project_id,
                Fact.subject_id == subject_id,
                Fact.status == FactStatus.active,
                (Fact.valid_to.is_(None)) | (Fact.valid_to > now),
                Fact.search_vector.op("@@")(entity_query),
            )
            .order_by(func.ts_rank_cd(Fact.search_vector, entity_query).desc())
            .limit(MULTI_HOP_MAX_PER_ENTITY)
        )
        for row in (await session.execute(stmt)).all():
            if row.id in seen:
                continue
            seen.add(row.id)
            found.append(row)
    return found


def _packet_fact(row: Any) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "predicate": row.predicate,
        "value": row.value,
        "confidence": row.confidence,
        "valid_from": row.valid_from.isoformat() if row.valid_from else None,
        "source_event_ids": [str(e) for e in row.source_event_ids],
    }


async def _open_conflict_fact_ids(
    session: AsyncSession, *, project_id: str, subject_id: str
) -> set[uuid.UUID]:
    stmt = select(ConflictSet).where(
        ConflictSet.project_id == project_id,
        ConflictSet.subject_id == subject_id,
        ConflictSet.status == "open",
    )
    ids: set[uuid.UUID] = set()
    for conflict in (await session.execute(stmt)).scalars().all():
        ids.update(conflict.fact_ids)
    return ids


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
    return {"facts": [], "episodes": [], "warnings": list(reasons), "status": "failed"}


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
) -> tuple[dict[str, Any], int, uuid.UUID]:
    """Assemble a ContextPacket. Returns (packet, token_count, trace_id).

    `extra_warnings` (e.g. policy warnings computed by the caller) are
    appended to the packet warnings BEFORE the trace is persisted, so the
    inspection trace shows exactly what the API returned.
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

    now = func.now()
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
    vector_top = (
        select(Fact.id)
        .where(*scope_filters)
        .order_by(Fact.embedding.cosine_distance(query_embedding))
        .limit(RETRIEVAL_TOP_K)
        .cte("vector_top")
    )
    fts_top = (
        select(Fact.id)
        .where(*scope_filters, Fact.search_vector.op("@@")(ts_query))
        .order_by(func.ts_rank_cd(Fact.search_vector, ts_query).desc())
        .limit(RETRIEVAL_TOP_K)
        .cte("fts_top")
    )
    candidates = select(vector_top.c.id).union(select(fts_top.c.id)).cte("candidates")

    # Phase 2 — full hybrid score on the candidate union only. Only the
    # columns needed for packing are selected: decoding the 384-dim embedding
    # of every returned row costs more than the scoring itself (measured in
    # the sprint-3 benchmark). The cap keeps the work flat no matter how many
    # facts the scope holds.
    stmt = (
        select(
            Fact.id,
            Fact.predicate,
            Fact.value,
            Fact.confidence,
            Fact.valid_from,
            Fact.source_event_ids,
            score.label("score"),
        )
        .where(Fact.id.in_(select(candidates.c.id)))
        .order_by(score.desc())
        .limit(CANDIDATE_LIMIT)
    )
    retrieval_start = time.perf_counter()
    rows = (await session.execute(stmt)).all()
    stage_timings["retrieval"] = round((time.perf_counter() - retrieval_start) * 1000)

    blocked_ids = await _open_conflict_fact_ids(
        session, project_id=project_id, subject_id=subject_id
    )

    decisions: list[dict[str, Any]] = []
    eligible: list[Any] = []
    for row in rows:
        if row.id in blocked_ids:
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "blocked",
                    "reason_code": "conflict_open",
                }
            )
        else:
            eligible.append(row)

    # Candidate facts waiting in an open conflict set are blocked too (they
    # are not active, so they never entered the scored pool).
    if blocked_ids:
        candidates = (
            (
                await session.execute(
                    select(Fact).where(
                        Fact.id.in_(blocked_ids),
                        Fact.status == FactStatus.candidate,
                    )
                )
            )
            .scalars()
            .all()
        )
        for fact in candidates:
            decisions.append(
                {
                    "fact_id": str(fact.id),
                    "action": "blocked",
                    "reason_code": "conflict_open",
                }
            )

    # Greedy packing under the token budget, best score first.
    packet_facts: list[dict[str, Any]] = []
    token_count = 0
    for row in eligible:
        cost = estimate_tokens(_render(row.predicate, row.value))
        if token_count + cost <= budget_tokens:
            packet_facts.append(_packet_fact(row))
            token_count += cost
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "included",
                    "reason_code": "top_score",
                }
            )
        else:
            decisions.append(
                {
                    "fact_id": str(row.id),
                    "action": "excluded",
                    "reason_code": "over_budget",
                }
            )

    # Multi-hop expansion: only worth trying if the main pass packed
    # something to seed entities from, and left room in the budget.
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
            exclude_ids=included_fact_ids | blocked_ids,
            now=now,
        )
        stage_timings["multi_hop_expansion"] = round(
            (time.perf_counter() - multi_hop_start) * 1000
        )
        for row in extra_rows:
            if token_count >= budget_tokens:
                break
            cost = estimate_tokens(_render(row.predicate, row.value))
            if token_count + cost <= budget_tokens:
                packet_facts.append(_packet_fact(row))
                token_count += cost
                decisions.append(
                    {
                        "fact_id": str(row.id),
                        "action": "included",
                        "reason_code": "multi_hop_expansion",
                    }
                )

    # Episodic memory (sprint 10): after the facts, the most relevant SOURCE
    # EVENTS of the same scope (cosine top-K over events.embedding, hnsw),
    # packed under the SAME budget — facts first, episodes with what
    # remains. This is what answers "what happened / when" questions: the
    # extractor keeps durable facts only, episodes keep the dated events.
    episodes_start = time.perf_counter()
    episode_rows = (
        (
            await session.execute(
                select(Event.id, Event.kind, Event.occurred_at, Event.payload)
                .where(
                    Event.project_id == project_id,
                    Event.subject_id == subject_id,
                    Event.embedding.is_not(None),
                )
                .order_by(Event.embedding.cosine_distance(query_embedding))
                .limit(EPISODE_TOP_K)
            )
        )
        .all()
    )
    stage_timings["episodes"] = round((time.perf_counter() - episodes_start) * 1000)
    packet_episodes: list[dict[str, Any]] = []
    for row in episode_rows:
        excerpt = episode_excerpt(row.kind, row.payload)
        cost = estimate_tokens(
            f"{row.occurred_at:%Y-%m-%d %H:%M} {row.kind} {excerpt}"
        )
        if token_count + cost <= budget_tokens:
            packet_episodes.append(
                {
                    "event_id": str(row.id),
                    "kind": row.kind,
                    "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
                    "excerpt": excerpt,
                }
            )
            token_count += cost
            decisions.append(
                {
                    "episode_id": str(row.id),
                    "action": "included",
                    "reason_code": "top_score",
                }
            )
        else:
            decisions.append(
                {
                    "episode_id": str(row.id),
                    "action": "excluded",
                    "reason_code": "over_budget",
                }
            )

    warnings: list[str] = []
    n_blocked = sum(1 for d in decisions if d["reason_code"] == "conflict_open")
    if n_blocked:
        warnings.append(
            f"open_conflict: {n_blocked} fact(s) hidden pending conflict resolution"
        )
    warnings.extend(extra_warnings or [])

    # Noisy-failure contract (ContextStatus): "degraded" whenever there is
    # something worth flagging (open conflicts, a caller-supplied warning
    # such as the missing-purpose policy notice). build_context never
    # produces "failed" itself — an internal failure raises (loud, same as
    # any other exception in this function); "failed" is for a CALLER that
    # catches that exception (see `failed_packet`).
    status = "degraded" if warnings else "ok"
    metrics.increment(f"context.{status}")

    packet = {
        "facts": packet_facts,
        "episodes": packet_episodes,
        "warnings": warnings,
        "status": status,
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
