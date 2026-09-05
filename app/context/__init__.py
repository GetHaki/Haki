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

Ranking (21 aout, see app.context.ranking): similarity (pgvector cosine)
and full-text rank (ts_rank_cd over the GENERATED search_vector column)
are min-max normalized over the query's own candidate pool, then combined
0.65/0.35. Recency (exponential decay, 30-day time constant, on
coalesce(valid_from, recorded_from) for facts / occurred_at for episodes)
is NOT part of that combination -- it is a tie-break only, applied after
relevance decides the order. HAKI_RANKING=legacy restores the pre-21-aout
raw weighted sum (0.6/0.25/0.15, recency included as a relevance term) as
a rollback path; see app.context.ranking for why that combination
under-weighted the lexical axis by roughly an order of magnitude in
practice and let recency dominate the ranking on off-topic queries.

Multi-hop expansion (sprint 10, deleted 23 aout): used to run a second
full-text-only lookup seeded by entities found in the facts just packed.
Measured before being rebuilt to compete in the unified pool: for every
question whose packet held part of its evidence, it found the missing turn
in 0 of 22 cases. Not underfunded, aimed at the wrong thing -- see the
block above the fragmentation detector in `build_context` for the full
measurement and why nothing replaces it.

Latency (sprint 3) — two-phase retrieval, standard candidate-generation +
rerank: scoring ALL active facts of a scope costs ~200 ms at 10k facts, so
phase 1 generates candidates with the INDEXES (top RETRIEVAL_TOP_K by hnsw
cosine distance UNION top RETRIEVAL_TOP_K by GIN full-text rank), and phase
2 computes the full hybrid score on that small union only (≤ 2×TOP_K rows),
then caps at CANDIDATE_LIMIT. Only the columns needed for packing are
selected: decoding the 1024-dim embedding of every returned row costs more
than the scoring itself (measured). Trade-off, documented: a fact that is
neither in the vector top-K nor in the full-text top-K cannot be served even
if recency would have lifted it — and facts beyond the cap are not traced.

Budget: tokens estimated as max(1, len(text) // 4). Facts and episodes are
packed from ONE ranked pool (key merging, 13 aout) — episodes ranked on
the exact same similarity/full-text/recency axes as facts, fused by
app.context.ranking (21 aout) — rather than two separately-budgeted pools
merged by a fixed share (the interim fix shipped 12 aout,
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

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Float, Text, cast, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from sqlalchemy.sql import func

from app import metrics
from app.config import settings
from app.context import cost as cost_module
from app.context.cost import (
    estimate_prose_tokens,
    estimate_tokens,
    render_line,
    short_timestamp,
)
from app.context.fts import build_query_tsquery
from app.context.ranking import legacy_weighted_sum, relevance
from app.errors import ApiError
from app.models import (
    ConflictSet,
    ContextTrace,
    EpisodeChunk,
    Event,
    Fact,
    FactStatus,
    SubjectAlias,
)
from app.providers import Embedder, Reranker, get_embedder, get_reranker

# Ranking mode names for settings.ranking (see app/config.py).
# "normalized" is the default and the only one that should be chosen on
# the merits; "legacy" is a rollback path, kept so that a ranking
# regression in production is one environment variable away from being
# undone. See app.context.ranking for the measurements and the rationale.
RANKING_NORMALIZED = "normalized"
RANKING_LEGACY = "legacy"

# Legacy scoring weights -- used ONLY by RANKING_LEGACY now (documented;
# never part of the public contract). The live weights live in
# app.context.ranking, where they apply to NORMALIZED axes and therefore
# mean what they say.
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

# Secondary sort keys for phase-1 candidate generation: they decide WHICH
# rows survive the LIMIT above whenever the primary key ties. Kept next to
# the K they bound, because the two only make sense together; see the block
# above the CTEs in build_context for why an id-only tie-break was not
# enough.
_FACT_TIEBREAK = (
    func.coalesce(Fact.valid_from, Fact.recorded_from).desc(),
    Fact.predicate,
    cast(Fact.value, Text),
    Fact.id,
)
_EPISODE_TIEBREAK = (
    EpisodeChunk.occurred_at.desc(),
    EpisodeChunk.ordinal.desc(),
    EpisodeChunk.id,
)

# Reranker (mechanism F-R, 15 aout): only the top RERANK_TOP_K candidates
# from the unified pool (by the existing hybrid score, facts and episodes
# combined) get a cross-encoder pass -- a cross-encoder scores one
# query-document PAIR per forward call, an order of magnitude slower than
# an embedding lookup, so running it over the FULL candidate set (up to
# CANDIDATE_LIMIT + EPISODE_TOP_K) would put a real latency cost in the hot
# path for no benefit: a candidate that did not even make the hybrid
# shortlist is vanishingly unlikely to be the one the cross-encoder should
# have promoted. 50 sits in the middle of the range cited in research/
# Haki_Livre_Construction_2026-08-15.md ("40-60 candidats RRF").
RERANK_TOP_K = 50

# M3 recall gate — floor on the SEMANTIC axis only (cosine distance of the
# candidate to the query), never on the hybrid score: similarity is the only
# bounded, embedder-calibratable term and carries the dominant weight (0.6);
# ts_rank_cd is unbounded, and even with the 'english' config (migration
# 0026) a shared content word alone can still yield a nonzero rank on a
# query where the semantic match is weak (a lexical escape hatch would let
# noise through); recency says nothing about relevance. Distinct from
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
# Episode candidates generated PER AXIS (top-K by vector distance, top-K
# by full-text rank; the pool is their union, so at most 2 x this).
#
# Was 8, when one candidate was one whole event: hydrating eight 4 000-
# character payloads was already the expensive part of the call. Since 21
# aout (migration 0027) a candidate is one turn-sized chunk, so eight of
# them is a handful of sentences -- far too narrow to rank over, given a
# single session produces ~20 chunks. Raised to RETRIEVAL_TOP_K so
# episodes and facts get the same candidate depth. The rows are small:
# 128 chunks of ~50 characters is less text than the 8 events this
# replaces.
EPISODE_TOP_K = RETRIEVAL_TOP_K

# Episode scoring (key merging, 13 aout; full-text axis added by mechanism
# E1a, 15 aout): facts and episodes are packed from ONE ranked pool, not
# two separate budgets — the previous fixed EPISODE_MIN_BUDGET_SHARE floor
# (12 aout) was an honest stopgap, not the real fix; this is the real fix.
# Episodes compute the exact same three axes as facts (similarity,
# full-text, recency) -- until 15 aout `events` had no search_vector
# column (facts did, migration 0004), so the lexical axis was structurally
# unavailable for episodes; migration 0022 gives episodes their own
# search_vector (events.index_text), closing that gap. Since 21 aout the
# two axes that matter for ranking (similarity, full-text) are combined by
# app.context.ranking on a pool that mixes facts and episodes together --
# no per-kind weights needed any more, see the fusion block in
# build_context.

# Episode budget ceiling (19 aout 2026): found via real measurement, not
# hypothesis. Raising budget_tokens 900->4000 on the same 458-question
# LoCoMo subset REGRESSED accuracy 31.4%->16.4%, not a plateau -- paired
# per-question diff showed 106 answers correct at the tighter budget
# became wrong at the wider one (eval/results/locomo_calibration_b4000_*).
# Root cause: once a larger budget_tokens admits big raw episodes into the
# unified score-ranked pool below, a single high-scoring episode can
# consume most of the extra room and displace several small, precise facts
# that fit comfortably in a tight budget -- worst for exact-value
# questions, the majority of LoCoMo. This is a CEILING on episodes' share
# of the budget, not the old EPISODE_MIN_BUDGET_SHARE floor (12 aout,
# removed 15 aout when facts/episodes were unified into one ranked pool):
# it never reserves space for episodes, it only stops them from crowding
# out facts once a wider budget lets more of them qualify. 0.5 reuses the
# exact value that floor used to hold -- a reasonable starting point, not
# re-derived; revisit once a proper sweep across budgets/shares is run.
EPISODE_MAX_BUDGET_SHARE = 0.5


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


def episode_row_excerpt(row: Any) -> str:
    """Verbatim text of one retrieved episode row.

    Since 21 aout (migration 0027) an episode row is an `episode_chunks`
    row, so the excerpt is the chunk's own verbatim slice rather than a
    re-serialisation of the whole parent payload. The cap stays: a single
    chunk is already bounded by app.context.chunking, and the token budget
    is what actually decides how much gets served.
    """
    return (row.text or "")[:EPISODE_EXCERPT_CHARS]


def _render(predicate: str, value: dict[str, Any]) -> str:
    return f"{predicate} {json.dumps(value, sort_keys=True, ensure_ascii=False)}"


# Capitalised-token entity detection, rule-based and LLM-free. Written
# for the multi-hop expansion, and outliving it (deleted 23 aout, see
# the block in build_context): the entity-aware fact scoring below is
# the live user now -- it tells a fact explicitly about someone the
# query does not name.
_ENTITY_TOKEN_RE = re.compile(r"[A-ZÀ-Ý][a-zà-ÿ]{2,}")
# Capitalized words that are common sentence-starters, not proper nouns —
# excluded so the heuristic doesn't seed expansion on noise.
_ENTITY_STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "here", "and", "with",
    "when", "where", "what", "which", "who",
    "le", "la", "les", "un", "une", "des", "ce", "cette", "ces", "et", "avec",
}


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
#
# Recalibrated by measurement (22 aout). Neither magnitude had ever been
# measured; both were wrong. On eval.retrieval_bench (LoCoMo conversations
# 1-2, n=231, budget 2000, gold served -- reproducible to the packet since
# the determinism fix of the same day):
#
#     boost 1.3 / penalty 0.3   (as shipped)      86.6 %
#     boost 1.3 / penalty 1.0                     86.6 %
#     boost 1.0 / penalty 1.0   (mechanism off)   87.9 %
#     boost 1.0 / penalty 0.85                    88.3 %
#     boost 1.0 / penalty 0.7   (this)            88.3 %
#     boost 1.0 / penalty 0.5                     88.3 %
#
# The BOOST is what cost the points: -1.3 on its own, the largest single
# term in the table. Promoting a fact because it carries the queried name
# pushes it above candidates that are simply more relevant, and a person's
# name in a question is a FILTER, not a relevance signal. It is gone --
# 1.0, a no-op -- rather than merely reduced, because there is no evidence
# for any promotion at all.
#
# The PENALTY earns its 0.4 point and is flat from 0.5 to 0.85, so this is
# not knife-edge calibration. 0.3 was not a re-ranking at all: on a score
# bounded in [0, 1] it is a near-exclusion, which is the opposite of what
# the paragraph above says the mechanism wants.
ENTITY_MATCH_BOOST = 1.0
ENTITY_MISMATCH_PENALTY = 0.7


def _query_entities(query: str) -> set[str]:
    return {t for t in _ENTITY_TOKEN_RE.findall(query) if t.lower() not in _ENTITY_STOPWORDS}


def _content_tiebreak(kind: str, row: Any) -> tuple[float, str]:
    """What separates two rows that the score and the recency cannot.

    Uniform types across both kinds, because facts and episodes are sorted in
    ONE pool: a float first, then a string. For an episode the later turn
    wins (negated ordinal) -- the same preference recency expresses one key
    earlier, and one that recency cannot express here since every chunk of a
    session shares its `occurred_at`. A fact has no such order, so its key is
    purely its content, and content is the point: it does not change when the
    same corpus is ingested again.
    """
    if kind == "episode":
        return (-float(getattr(row, "ordinal", 0) or 0), str(getattr(row, "text", "")))
    content = (
        f"{getattr(row, 'predicate', '')}\x00"
        f"{json.dumps(getattr(row, 'value', None), sort_keys=True, default=str)}"
    )
    return (0.0, content)


def _entity_adjusted_score(score: float, value: Any, query_entities: set[str]) -> float:
    """Demote a fact explicitly about someone the query does not name.

    Matching is TOKEN OVERLAP, not string equality (22 aout). The previous
    rule compared a whole tagged name against `query_entities`, a set that
    only ever holds SINGLE capitalised tokens because that is what
    `_query_entities` extracts -- so a fact tagged "John Smith" could not
    match any query, ever, and was demoted as if it were about somebody
    else. That is precisely the failure this mechanism exists to prevent,
    inverted.

    A name too short for `_ENTITY_TOKEN_RE` (it needs a capital and two
    lowercase letters) tokenises to nothing and falls back to equality, so
    overlap is a strict superset of the old rule: it can add a match, never
    remove one. Instrumented over the whole retrieval bench the two rules
    disagreed on ZERO candidate/query pairs -- every LoCoMo speaker name is
    one token, and a one-token name tokenises to itself -- so the bench
    could not decide this and did not have to.
    """
    if not query_entities or not isinstance(value, dict):
        return score
    person = value.get("person")
    if not isinstance(person, str) or not person:
        return score
    tokens = set(_ENTITY_TOKEN_RE.findall(person))
    matched = bool(tokens & query_entities) if tokens else person in query_entities
    return score * (ENTITY_MATCH_BOOST if matched else ENTITY_MISMATCH_PENALTY)


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


def _relative_to_now(dt: datetime, now: datetime) -> str:
    """Render `dt` as an exact, precomputed offset from `now` (mechanism
    F1, 15 aout): "N days/weeks before/after the question". Deterministic
    Python arithmetic, not left for the reader to derive — the point of
    dual-date rendering (Partie 3.6): the reader VERIFIES a number that is
    already in the text instead of calculating one from two ISO dates,
    which a gpt-4o-mini-class reader gets right only 13.5-16% of the time
    (Test-of-Time Arithmetic). Exact day count always shown; grouped into
    weeks/months too when that reads better, but never in place of the
    exact count.
    """
    days = round((now - dt).total_seconds() / 86400)
    if days == 0:
        return "the day of the question"
    direction = "before" if days > 0 else "after"
    n = abs(days)
    if n >= 60:
        return f"{n} days (~{round(n / 30)} months) {direction} the question"
    if n >= 14 and n % 7 == 0:
        return f"{n} days ({n // 7} weeks) {direction} the question"
    return f"{n} day{'s' if n != 1 else ''} {direction} the question"


def _packet_episode(
    row: Any,
    excerpt: str,
    now: datetime | None,
    *,
    ref: str,
    context_neighbor: bool = False,
) -> dict[str, Any]:
    """One served turn, in the shape the packet exposes.

    Was built inline in the packing loop; costed from its rendered line
    now (22 aout), and a line needs a `ref`, so it became its own function.
    """
    occurred_at = row.occurred_at.isoformat() if row.occurred_at else None
    return {
        # The PARENT event id, not the chunk id: it is what /v1/timeline
        # and /v1/inspect address, and what the packet has always exposed.
        # The chunk id stays internal to ranking and tracing.
        "event_id": str(row.event_id),
        "episode_id": str(row.id),
        "ref": ref,
        "kind": row.kind,
        "occurred_at": occurred_at,
        "occurred_at_short": short_timestamp(occurred_at),
        # Dual-date rendering (mechanism F1, 15 aout) -- see
        # _relative_to_now / PacketFact.valid_from_relative.
        "occurred_at_relative": (
            _relative_to_now(row.occurred_at, now) if row.occurred_at else None
        ),
        "excerpt": excerpt,
        "context_neighbor": context_neighbor,
    }


def _packet_fact(
    row: Any,
    freshness: str = "current",
    *,
    conflict_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    reference = row.last_reinforced_at or row.valid_from or row.recorded_from
    return {
        "id": str(row.id),
        "predicate": row.predicate,
        "value": row.value,
        "confidence": row.confidence,
        "valid_from": row.valid_from.isoformat() if row.valid_from else None,
        # Same instant, trimmed of the seconds and the UTC offset nothing
        # reads (22 aout) -- see app.context.cost.short_timestamp. Additive:
        # an SDK that does not know this field renders `valid_from` as before.
        "valid_from_short": short_timestamp(
            row.valid_from.isoformat() if row.valid_from else None
        ),
        # Dual-date rendering (mechanism F1, 15 aout): exact offset from the
        # temporal point of view (`as_of`, defaulting to real "now" -- see
        # `now_dt` in build_context), so the reader reads a number instead
        # of computing one. None only when valid_from itself is None.
        "valid_from_relative": (
            _relative_to_now(row.valid_from, now) if row.valid_from and now else None
        ),
        # End of the validity interval (Bench-2): when a supersession set
        # this, the reader can see that a value STOPPED being true instead
        # of guessing from two "valid from" dates. Additive -- readers that
        # do not know this field keep using valid_from exactly as before.
        "valid_to": (
            row.valid_to.isoformat() if getattr(row, "valid_to", None) else None
        ),
        # Identity qualifiers (Bench-2): the condition this fact holds under
        # (team, person, ...). The answer prompt groups facts "by what they
        # are actually about" -- without these, two facts under one
        # predicate read as one contradicting pair instead of two scoped
        # truths. Additive, same contract as valid_to above.
        "qualifiers": dict(getattr(row, "qualifiers", None) or {}),
        # When this fact's source text used a relative time expression
        # ("last week"), the ISO range the extractor resolved it to,
        # anchored on the source event's occurred_at -- distinct from
        # valid_from (always the MESSAGE's own timestamp). See
        # app.providers.base.ExtractedFact.temporal_range.
        "temporal_range": getattr(row, "temporal_range", None),
        # When the fact is ABOUT (21 aout), as a normalised instant --
        # distinct from valid_from, which is when it was SAID. A reader
        # answering "when did X happen?" had to pick it out of free-form
        # `value` JSON or out of `temporal_range`, in two different shapes;
        # this is the same information in one, always the same, place.
        # None for the many facts that are about no particular instant.
        "observed_at": (
            row.observed_at.isoformat() if getattr(row, "observed_at", None) else None
        ),
        "observed_at_relative": (
            _relative_to_now(row.observed_at, now)
            if getattr(row, "observed_at", None) and now
            else None
        ),
        # Reclassification safety net (found by code review, 16 aout): True
        # when this fact was activated by the automatic overflow
        # reclassification (a 3rd competing "state" value flipping the
        # whole identity to "event") rather than an extractor declaring
        # memory_form="event" up front. See Fact.reclassified_at -- never
        # hidden, always flagged, same honest-degradation contract as
        # "contested"/"unconfirmed"/"stale" above.
        "auto_reclassified": getattr(row, "reclassified_at", None) is not None,
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
    budget_tokens: int = 3000,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    extra_warnings: list[str] | None = None,
    as_of: datetime | None = None,
    exclude_ids: list[str] | None = None,
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

    `reranker` (mechanism F-R, 15 aout, Sprint 2): only consulted, and only
    ever instantiated, when `settings.rerank_enabled` -- omitted here, an
    unset flag means the cross-encoder pass never runs and every candidate
    keeps its hybrid-formula score exactly as before this mechanism
    existed. See RERANK_TOP_K below for what actually gets reranked.

    `budget_tokens` default (3000 since 22 aout, when the budget started
    counting the line the caller actually receives -- see
    app.schemas.context.ContextRequest for the measured curve): external
    accuracy-vs-budget curves cited in research/Haki_Livre_Construction_
    2026-08-15.md agree the gain from more served context flattens well
    before 4000 tokens with a gpt-4o-mini-class reader -- ~1500-2500 is
    the cited working point, not "more is free" (one curve even shows
    MORE context making an already-weak reader WORSE).
    """
    if budget_tokens <= 0:
        raise ApiError(
            type="budget_exceeded",
            message="budget_tokens must be a positive integer",
            field="budget_tokens",
        )
    embedder = embedder or get_embedder()
    # Reranker (mechanism F-R): never instantiated unless the flag is set
    # AND no explicit reranker was already passed by the caller (tests pass
    # FakeProvider directly regardless of the flag).
    if settings.rerank_enabled and reranker is None:
        reranker = get_reranker()
    request_start = time.perf_counter()
    stage_timings: dict[str, int] = {}

    embed_start = time.perf_counter()
    # embed_query, not embed: see app.providers.base.Embedder. getattr
    # rather than a hard call so an embedder written before this split (a
    # self-hosted custom provider) keeps working -- for those, symmetric is
    # both the previous behaviour and the safe assumption.
    embed_query = getattr(embedder, "embed_query", embedder.embed)
    # Perf-1: the embedding runs in a worker thread (local.py) or over the
    # network (remote providers) and never touches `session`, so it overlaps
    # with the tsquery lexeme SELECT below instead of preceding it. The
    # facts/episodes/conflict SELECTs further down stay sequential on
    # purpose: an AsyncSession is a single connection and does not support
    # concurrent queries, and opening parallel sessions would lose the
    # request's RLS context (get_session sets haki.project_id per session).
    # True branch parallelism needs RLS propagation first -- tracked, not
    # attempted here.
    query_embedding, ts_query_pair = await asyncio.gather(
        embed_query([query]),
        build_query_tsquery(session, query),
    )
    query_embedding = query_embedding[0]
    ts_query, _ts_query_text = ts_query_pair
    stage_timings["embed"] = round((time.perf_counter() - embed_start) * 1000)

    # A bound literal (not just a Python datetime) so it can take part in
    # SQL arithmetic (`now - Fact.valid_from`) exactly like func.now() does.
    now = literal(as_of) if as_of is not None else func.now()
    similarity = func.coalesce(1 - Fact.embedding.cosine_distance(query_embedding), 0.0)
    # Lexical axis (20 aout, upgraded same day): an OR tsquery built from
    # the query's own lexemes, in the configuration the search_vector
    # columns were GENERATED with -- see app.context.fts for the full
    # rationale and the verified measurements. websearch_to_tsquery joins
    # terms with AND, which -- combined with even a stemmed/stopword-aware
    # config -- still requires every remaining content word to be present;
    # the OR form lets ranking, not a filter, decide relevance. Matching
    # facts.search_vector / events.search_vector's own to_tsvector config
    # is still required, or the query and column lexemes stop lining up
    # and NO match ever fires (app.db.verify_fts_config guards this at
    # startup).
    # The debug text (the assembled tsquery, e.g. "'carolin' | 'go'") is
    # not wired into the context trace yet -- a real but separate piece of
    # work (a new ContextTrace column + migration). Discarded here rather
    # than left unused (the tsquery itself was built above, overlapping
    # the embedding).
    fulltext = func.coalesce(func.ts_rank_cd(Fact.search_vector, ts_query), 0.0)
    # `greatest(..., 0)` (20 aout, bug): without it, a fact whose
    # valid_from is AFTER the point of view (`as_of`, always set by the
    # eval harness) would make the exponent POSITIVE -- exp() returning
    # 2.72 at 30 days ahead, 20 at 90 days, instead of a value in [0, 1].
    # With W_RECENCY = 0.15 that single term would outweigh similarity and
    # full-text combined and pin future-dated facts at the top of the
    # ranking. The scope filter below (valid_from <= now) already removes
    # those facts outright; this clamp is the second line of defence for
    # rows it does not cover (recorded_from NULL edge cases).
    recency = func.exp(
        -func.greatest(
            func.extract(
                "epoch", now - func.coalesce(Fact.valid_from, Fact.recorded_from)
            ),
            0.0,
        )
        / RECENCY_TAU_SECONDS
    )
    # No composite score in SQL any more (21 aout): the three axes are
    # returned as they are and combined in Python by app.context.ranking,
    # so that facts and episodes -- two separate queries -- are ranked
    # against each other on ONE consistent scale, and so that the
    # combination is unit-testable without a database.

    # Hard filters first: exact scope, active only, still valid.
    #
    # The lower bound (valid_from <= now) matters for more than correctness
    # of the window: without it, a fact whose effective date sits after
    # `now`/`as_of` doesn't just leak into scope early -- the recency term
    # above, exp(-(now - valid_from) / TAU), goes POSITIVE for that negative
    # elapsed time (exponential growth instead of decay), so the fact's
    # score can run away and crowd out genuinely relevant facts from the
    # packet. Found via external code audit, confirmed 20 aout 2026.
    # Items the caller already holds from an earlier packet (23 aout).
    # Applied at CANDIDATE GENERATION, not after packing: excluding them
    # later would still spend the top-K slots on rows the caller is going
    # to throw away.
    #
    # This is "the next page", and it is deliberately not called anything
    # grander. Measured on the questions whose first packet holds part of
    # their evidence: re-asking the SAME question with the seen items
    # excluded finds the missing turn 44.8 % of the time. Re-asking with a
    # query reformulated from what the first packet contained finds it
    # 41.4 % of the time, and asking with that content alone -- the most
    # generous form, the exact text an agent just read -- 27.6 %. So this
    # does not do multi-hop, and the documentation must not say it does:
    # what it does is serve further down the same ranked list, which is
    # the same thing as a larger budget, paid only by the callers who
    # turn out to need it.
    excluded: set[uuid.UUID] = set()
    for raw in exclude_ids or ():
        try:
            excluded.add(uuid.UUID(str(raw)))
        except (ValueError, AttributeError, TypeError):
            # A malformed id is the caller's typo, not a reason to fail a
            # memory read: the packet is still correct, it just holds an
            # item they meant to skip.
            continue

    scope_filters = [
        Fact.project_id == project_id,
        Fact.subject_id == subject_id,
        Fact.status == FactStatus.active,
        (Fact.valid_to.is_(None)) | (Fact.valid_to > now),
        func.coalesce(Fact.valid_from, Fact.recorded_from) <= now,
    ]
    if excluded:
        scope_filters.append(Fact.id.not_in(excluded))

    # Phase 1 — candidate generation with the indexes (hnsw + GIN). Without
    # this, phase 2 would score every active fact of the scope (~200 ms at
    # 10k facts in the sprint-3 benchmark).
    #
    # Every ORDER BY below carries `_FACT_TIEBREAK` as a secondary key, not
    # a bare `Fact.id` (22 aout). A random id was still A key -- ties on the
    # primary axis never crashed anything -- but it is a random one: two
    # otherwise-identical calls that both hit a tied group can keep a
    # DIFFERENT subset of it, because which uuid4 sorts first has nothing to
    # do with the facts themselves. Ingest the same corpus twice and the two
    # installs' packets can differ purely from that (confirmed: scripts/
    # check_retrieval_discrimination.py, run twice against the same
    # project/subject/query/budget, returned two DIFFERENT fact sets).
    # `_FACT_TIEBREAK` breaks ties on recency then content instead, so the
    # same facts always win the same LIMIT.
    vector_top = (
        select(Fact.id)
        .where(*scope_filters)
        .order_by(Fact.embedding.cosine_distance(query_embedding), *_FACT_TIEBREAK)
        .limit(RETRIEVAL_TOP_K)
        .cte("vector_top")
    )
    fts_top = (
        select(Fact.id)
        .where(*scope_filters, Fact.search_vector.op("@@")(ts_query))
        .order_by(func.ts_rank_cd(Fact.search_vector, ts_query).desc(), *_FACT_TIEBREAK)
        .limit(RETRIEVAL_TOP_K)
        .cte("fts_top")
    )
    candidates = select(vector_top.c.id).union(select(fts_top.c.id)).cte("candidates")

    # Phase 2 — full hybrid score on the candidate union only. Only the
    # columns needed for packing are selected: decoding the 1024-dim embedding
    # of every returned row costs more than the scoring itself (measured in
    # the sprint-3 benchmark). The cap keeps the work flat no matter how many
    # facts the scope holds.
    #
    # `Fact.id` as a secondary sort key (13 aout, "Bug 2" diagnostic, 11
    # aout): facts written in the same consolidation batch routinely share
    # their recency value down to the minute, so for an off-topic query a
    # whole batch can tie exactly on every axis. Without a stable secondary
    # key, Postgres's query plan is free to return a tied group in a
    # different order between two otherwise-identical calls -- confirmed
    # empirically (scripts/check_retrieval_discrimination.py): the same
    # project, subject, query and budget returned two different fact sets
    # on consecutive runs, purely from tie order. This was the ACTUAL
    # mechanism behind the originally-reported "Bug 2" symptom (five
    # different questions returning an identical packet) once budget
    # headroom alone was ruled out. This ORDER BY is generation-stage only
    # now (21 aout) -- the same reasoning applies, more directly, to the
    # final fusion sort below (`pool.sort`, id as its own last tie-break).
    stmt = (
        select(
            Fact.id,
            Fact.predicate,
            Fact.value,
            Fact.confidence,
            Fact.valid_from,
            Fact.valid_to,
            Fact.source_event_ids,
            Fact.recorded_from,
            Fact.last_reinforced_at,
            Fact.fact_kind,
            Fact.volatility,
            Fact.origin_trust,
            Fact.qualifiers,
            Fact.temporal_range,
            Fact.reclassified_at,
            # When the fact is ABOUT, normalised (migration 0029) -- see
            # _packet_fact for the rendering.
            Fact.observed_at,
            # Exact fact-to-turn provenance (migration 0028). Used by the
            # context window below to serve the turn a fact was actually
            # extracted from, instead of guessing among the turns of its
            # source event.
            Fact.source_chunk_id,
            Fact.embedding.cosine_distance(query_embedding).label("distance"),
            similarity.cast(Float).label("similarity"),
            fulltext.cast(Float).label("fulltext"),
            recency.cast(Float).label("recency"),
        )
        .where(Fact.id.in_(select(candidates.c.id)))
        # Generation-stage ordering only. `candidates` is the union of two
        # LIMIT RETRIEVAL_TOP_K CTEs, so it holds at most 2 x
        # RETRIEVAL_TOP_K ids and CANDIDATE_LIMIT cannot bite; the final
        # ordering is the fusion below, not this ORDER BY. `Fact.id` keeps
        # it deterministic anyway -- exact ties on an axis are routine
        # here (see the tie-break note on the fusion sort below).
        .order_by(similarity.desc(), Fact.id)
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

    # Every member of a genuine 2-member disagreement is hydrated once, even
    # the losing side (still `candidate`, never scored by the phase-2 query
    # above) — so it can be packed alongside its active sibling. But only
    # for a conflict TOUCHING this query's own `rows`: a conflict with no
    # member in `rows` cannot produce a served contested fact here (the
    # packing loop below only ever looks up siblings of a packed `rows`
    # member), so hydrating its members is pure work. This keeps the
    # hydrate proportional to served facts, not to open conflicts.
    row_ids = {row.id for row in rows}
    served_contested_fact_ids: set[uuid.UUID] = set()
    _seen_conflicts: set[int] = set()
    for fid, conflict in contested_conflict_by_fact.items():
        if id(conflict) in _seen_conflicts:
            continue
        _seen_conflicts.add(id(conflict))
        if any(member in row_ids for member in conflict.fact_ids):
            served_contested_fact_ids.update(conflict.fact_ids)
    contested_rows_by_id: dict[uuid.UUID, Fact] = {}
    if served_contested_fact_ids:
        # Same rule as the phase-2 query above: never decode the 1024-dim
        # embedding or the tsvector for rows the packet renders from light
        # columns only (_packet_fact + _fact_freshness touch neither).
        contested_members = (
            (
                await session.execute(
                    select(Fact)
                    .options(defer(Fact.embedding), defer(Fact.search_vector))
                    .where(
                        Fact.id.in_(served_contested_fact_ids)
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
        # Audit-only rows: the trace needs the id, nothing else. Selecting
        # the full entity would decode the 1024-dim embedding per row for
        # a decision that carries no content.
        quarantined_candidate_ids = (
            (
                await session.execute(
                    select(Fact.id).where(
                        Fact.id.in_(quarantined_ids),
                        Fact.status == FactStatus.candidate,
                    )
                )
            )
            .scalars()
            .all()
        )
        for fact_id in quarantined_candidate_ids:
            decisions.append(
                {
                    "fact_id": str(fact_id),
                    "action": "blocked",
                    "reason_code": "conflict_open",
                }
            )

    # Episodic memory (sprint 10, key merging 13 aout): the most relevant
    # turn-sized SLICES of the same scope, on the SAME three axes as facts
    # (similarity, full-text, recency), so they compete fairly in ONE
    # ranked pool below (app.context.ranking, 21 aout), not two
    # separately-budgeted ones. This is what answers "what happened /
    # when" questions, and (12-13 aout finding) carries information a
    # compact fact loses entirely: the extractor keeps durable facts only,
    # episodes keep the dated events in their own words.
    #
    # Episodes are read from `episode_chunks`, not from `events` (21 aout,
    # migration 0027): one turn-sized slice per row instead of one whole
    # payload. An event was costing 810 tokens of a 900-token budget to
    # serve and had 87.6 % of itself outside the embedder's ~128-token
    # window -- see app.context.chunking for the measurements. `events`
    # remains the ledger; this table is derived from it and rebuildable at
    # any time.
    episode_similarity = func.coalesce(
        1 - EpisodeChunk.embedding.cosine_distance(query_embedding), 0.0
    )
    # Full-text axis (mechanism E1a, 15 aout): the same ts_query already
    # built for facts above -- one lexical query, several search_vector
    # columns. NULL-safe: a chunk whose embedding or index_text is missing
    # scores 0 on the axis rather than dropping out of the pool.
    episode_fulltext = func.coalesce(
        func.ts_rank_cd(EpisodeChunk.search_vector, ts_query), 0.0
    )
    # Same clamp as the fact recency above (20 aout): a chunk dated after
    # the point of view would otherwise score exp(+x). The `occurred_at <=
    # now` filter below is the primary fix; this keeps the expression
    # well-defined regardless.
    episode_recency = func.exp(
        -func.greatest(func.extract("epoch", now - EpisodeChunk.occurred_at), 0.0)
        / RECENCY_TAU_SECONDS
    )
    episodes_start = time.perf_counter()
    # Candidate generation mirrors the two-CTE shape facts have had since
    # sprint 3: top-K by vector distance UNION top-K by full-text rank, on
    # IDS ONLY. Before 21 aout episodes were generated by the composite
    # score itself, so a broken composite silently broke generation too --
    # an episode only the lexical axis could find never entered the pool,
    # whatever the ranking did afterwards.
    episode_scope = (
        EpisodeChunk.project_id == project_id,
        EpisodeChunk.subject_id == subject_id,
        EpisodeChunk.embedding.is_not(None),
        # Point of view (20 aout): a chunk of a conversation that happens
        # after `as_of` has not happened yet as far as this call is
        # concerned.
        EpisodeChunk.occurred_at <= now,
        # Provenance guard (M8): untrusted-origin content is never served
        # as an episode -- an episode is a VERBATIM excerpt replayed into
        # the agent's context, i.e. a direct injection channel. Facts
        # extracted from it already go through the quarantine path; the raw
        # text must not bypass it. Denormalised onto the chunk so this
        # filter stays index-friendly.
        EpisodeChunk.origin_trust != "untrusted",
    )
    if excluded:
        # Matched on the chunk id, which is what the packet exposes as
        # `episode_id`. An `event_id` passed here excludes nothing, on
        # purpose: it names a whole session, and dropping every turn of it
        # because one was served is not what "I already have this" means.
        episode_scope = (*episode_scope, EpisodeChunk.id.not_in(excluded))
    episode_vector_top = (
        select(EpisodeChunk.id)
        .where(*episode_scope)
        .order_by(
            EpisodeChunk.embedding.cosine_distance(query_embedding), *_EPISODE_TIEBREAK
        )
        .limit(EPISODE_TOP_K)
        .cte("episode_vector_top")
    )
    episode_fts_top = (
        select(EpisodeChunk.id)
        .where(*episode_scope, EpisodeChunk.search_vector.op("@@")(ts_query))
        .order_by(
            func.ts_rank_cd(EpisodeChunk.search_vector, ts_query).desc(), *_EPISODE_TIEBREAK
        )
        .limit(EPISODE_TOP_K)
        .cte("episode_fts_top")
    )
    episode_candidates = (
        select(episode_vector_top.c.id)
        .union(select(episode_fts_top.c.id))
        .cte("episode_candidates")
    )
    episode_rows = (
        (
            await session.execute(
                select(
                    EpisodeChunk.id,
                    EpisodeChunk.event_id,
                    EpisodeChunk.ordinal,
                    EpisodeChunk.occurred_at,
                    EpisodeChunk.text,
                    Event.kind,
                    EpisodeChunk.embedding.cosine_distance(query_embedding).label(
                        "distance"
                    ),
                    episode_similarity.cast(Float).label("similarity"),
                    episode_fulltext.cast(Float).label("fulltext"),
                    episode_recency.cast(Float).label("recency"),
                )
                .join(Event, Event.id == EpisodeChunk.event_id)
                .where(EpisodeChunk.id.in_(select(episode_candidates.c.id)))
                # Generation-stage ordering, like the facts query above: the
                # union of two LIMIT EPISODE_TOP_K CTEs holds at most
                # 2 x EPISODE_TOP_K ids. The final ordering is the ranking.
                .order_by(episode_similarity.desc(), EpisodeChunk.id)
            )
        )
        .all()
    )
    stage_timings["episodes"] = round((time.perf_counter() - episodes_start) * 1000)

    # Unified ranked pool (key merging): facts and episodes, both filtered
    # by the SAME recall floor (M3, semantic/distance axis only -- never
    # the relevance score, see RECOMMENDED_RECALL_MAX_DISTANCE's comment),
    # then ranked together (fusion below) and packed greedily against
    # budget_tokens in a SINGLE pass. A fact and an episode compete on
    # their actual merits now, not on which separately-budgeted pool they
    # happened to land in.
    recall_max_distance = settings.recall_max_distance
    query_entities = _query_entities(query)
    survivors: list[tuple[str, Any]] = []
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
        survivors.append(("fact", row))
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
        survivors.append(("episode", row))

    # Ordering (21 aout). Two changes, both measured -- see
    # app.context.ranking for the numbers and for why RRF, tried first,
    # was rejected:
    #
    #   * the two relevance axes are min-max normalized over THIS query's
    #     pool before being combined, so a fact's ts_rank_cd (over ~50
    #     characters of search_text) and an episode's (over up to 4 000)
    #     become comparable, and the declared weights are the applied ones;
    #   * recency leaves the relevance score entirely and becomes a
    #     tie-break. It was never a relevance signal: weighted 0.15 against
    #     a similarity spread of ~0.1, it pulled the most recent candidates
    #     to the top whatever the query, and they then ate the budget. As a
    #     tie-break it still does the one job it was wanted for -- between
    #     two candidates the query cannot separate, the newer one wins.
    #
    # Entity affinity stays a multiplier on the relevance score, unchanged:
    # the score it multiplies is now bounded in [0, 1], which is the scale
    # ENTITY_MATCH_BOOST/ENTITY_MISMATCH_PENALTY were calibrated against in
    # the first place.
    pool: list[tuple[float, str, Any]] = []
    tiebreak: dict[int, float] = {}
    if survivors:
        similarity_axis = [float(row.similarity or 0.0) for _, row in survivors]
        fulltext_axis = [float(row.fulltext or 0.0) for _, row in survivors]
        recency_axis = [float(row.recency or 0.0) for _, row in survivors]
        if settings.ranking == RANKING_LEGACY:
            scores = legacy_weighted_sum(
                similarity_axis,
                fulltext_axis,
                recency_axis,
                (W_SIMILARITY, W_FULLTEXT, W_RECENCY),
            )
        else:
            scores = relevance(similarity_axis, fulltext_axis)
        scores = [
            _entity_adjusted_score(score, getattr(row, "value", None), query_entities)
            for score, (_, row) in zip(scores, survivors)
        ]
        pool = [(score, kind, row) for score, (kind, row) in zip(scores, survivors)]
        tiebreak = {
            id(row): recency_value
            for (_, row), recency_value in zip(survivors, recency_axis)
        }
    # Sort key, in order: relevance, then recency, then CONTENT, then id
    # (22 aout, added the content key). `row.id` alone used to be the last
    # tie-break -- not decoration, exact ties are routine (a low-signal
    # query ties a whole consolidation batch) and Postgres gives no stable
    # order for tied rows between two identical calls, confirmed
    # empirically in scripts/check_retrieval_discrimination.py -- but a
    # uuid4 is RANDOM: which of two tied rows keeps the last budget slot
    # then depended on nothing about the rows themselves, so the same
    # corpus ingested twice could keep a different subset of a tied group.
    # `_content_tiebreak` sorts on what the fact/episode actually says
    # first; `str(item[2].id)` only remains to make the order total when
    # even the content is identical.
    pool.sort(
        key=lambda item: (
            -item[0],
            -tiebreak.get(id(item[2]), 0.0),
            *_content_tiebreak(item[1], item[2]),
            str(item[2].id),
        )
    )

    # Reranker (mechanism F-R, 15 aout): re-order only the top RERANK_TOP_K
    # candidates (already sorted by the hybrid score above) with a
    # cross-encoder pass, then keep that block strictly ahead of the
    # untouched tail. Cross-encoder scores are NOT on the same scale as the
    # hybrid formula's -- re-sorting the two blocks independently and
    # concatenating (rather than merging both into one global sort by raw
    # score value) is what keeps that safe: every reranked candidate still
    # outranks every un-reranked one, exactly as it did before reranking
    # touched anything, only the ORDER within the shortlist changes.
    if reranker is not None and pool:
        rerank_start = time.perf_counter()
        head_items = pool[:RERANK_TOP_K]
        tail = pool[RERANK_TOP_K:]
        documents = [
            _render(row.predicate, row.value) if kind == "fact" else episode_row_excerpt(row)
            for _score, kind, row in head_items
        ]
        rerank_scores = await reranker.rerank(query, documents)
        reranked_head = sorted(
            (
                (new_score, kind, row)
                for new_score, (_old_score, kind, row) in zip(rerank_scores, head_items)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        pool = reranked_head + tail
        stage_timings["rerank"] = round((time.perf_counter() - rerank_start) * 1000)

    packet_facts: list[dict[str, Any]] = []
    packet_episodes: list[dict[str, Any]] = []
    packed_fact_ids: set[uuid.UUID] = set()
    token_count = 0
    episode_token_count = 0
    episode_budget = budget_tokens * EPISODE_MAX_BUDGET_SHARE
    episodes_packed = 0
    # Whether a CONTESTED marker was actually charged this call, for
    # overhead_tokens below (22 aout): the chain-of-note paragraph is only
    # worth paying for when a conflict is genuinely being served.
    contested_charged = False
    # The item is BUILT first, then costed from the line it will actually
    # render as (22 aout, app.context.cost). Costing a stripped
    # `predicate value` string instead put a median of 4 565 tokens into the
    # caller's prompt for budget_tokens=2000 -- 2.28x what they asked for,
    # on every call. `ref` is assigned here because it is part of that line;
    # an index that ends up unused when the item does not fit is simply
    # taken by the next one.
    for _score, kind, row in pool:
        if kind == "fact":
            conflict = contested_conflict_by_fact.get(row.id)
            item = _packet_fact(
                row,
                freshness_by_id.get(row.id, "current"),
                conflict_id=str(conflict.id) if conflict else None,
                now=now_dt,
            )
            item["ref"] = f"F{len(packet_facts) + 1}"
        else:
            excerpt = episode_row_excerpt(row)
            item = _packet_episode(row, excerpt, now_dt, ref=f"E{len(packet_episodes) + 1}")
        item["line"] = render_line(kind, item)
        cost = estimate_tokens(item["line"])
        if kind == "episode" and episodes_packed > 0 and episode_token_count + cost > episode_budget:
            # Ceiling hit -- this episode would eat into facts' share of the
            # budget. Never blocks the single best-ranked episode (that one
            # is still worth serving on its own merits at any budget size,
            # same as before this ceiling existed) -- only kicks in once a
            # SECOND or later episode would pile on top of it. Skip (not
            # break: lower-ranked facts further down the pool still deserve
            # a shot at the room this leaves free) rather than count it
            # against the total budget at all.
            decisions.append(
                {"episode_id": str(row.id), "action": "excluded", "reason_code": "over_episode_budget"}
            )
            continue
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
        if kind == "episode":
            episode_token_count += cost
            episodes_packed += 1
        if kind == "fact":
            if conflict is not None:
                contested_charged = True
            packet_facts.append(item)
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
                    if sibling.status in (FactStatus.disabled, FactStatus.deleted):
                        # B2: a forgotten member of an open conflict set is
                        # never served again, not even as a contested sibling.
                        # Traced so the inspect view stays auditable.
                        decisions.append(
                            {
                                "fact_id": str(sibling_id),
                                "action": "blocked",
                                "reason_code": "forgotten",
                            }
                        )
                        continue
                    sibling_freshness = _fact_freshness(sibling, now_dt)
                    sibling_item = _packet_fact(
                        sibling,
                        sibling_freshness,
                        conflict_id=str(conflict.id),
                        now=now_dt,
                    )
                    sibling_item["ref"] = f"F{len(packet_facts) + 1}"
                    sibling_item["line"] = render_line("fact", sibling_item)
                    sibling_cost = estimate_tokens(sibling_item["line"])
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
                    packet_facts.append(sibling_item)
                    packed_fact_ids.add(sibling_id)
                    decisions.append(
                        {
                            "fact_id": str(sibling_id),
                            "action": "included",
                            "reason_code": "conflict_disputed",
                        }
                    )
        else:
            packet_episodes.append(item)
            decisions.append(
                {"episode_id": str(row.id), "action": "included", "reason_code": "top_score"}
            )

    # The two bolt-ons that used to sit here are gone (23 aout). Both ran
    # "against whatever budget the unified pass left", which is a guarantee
    # of doing nothing: a greedy loop stops when the NEXT item does not fit,
    # so the leftover is by construction smaller than one line. Measured,
    # budget 3000, 231 questions: the expansion's trigger passed 91.8 % of
    # the time with room for one row in 0.0 % of them (11 tokens left on
    # average); the context window served 0 neighbours out of 8 653
    # episodes. Eight passing tests certified them, all at budgets of 45,
    # 200 and 900, where the main pool cannot fill the budget.
    #
    # MULTI-HOP EXPANSION: deleted, not repaired. Before rebuilding it I
    # measured whether it could work at all -- for each question whose
    # packet holds part of its evidence, take the gold turn that WAS served,
    # extract its entities exactly as _candidate_entities did, run the same
    # lexical query: does the missing turn come back? 0 times out of 22. It
    # was not starved, it was aimed at the wrong thing.
    #
    # CONTEXT WINDOW: rebuilt inside the packing loop, charged like
    # everything else, swept -- and then deleted, because it does not pay.
    # `any` / `complete` / multi-hop complete:
    #
    #     no neighbour                     88.3 / 75.8 / 32.6
    #     top 3 anchors carry one          89.2 / 76.6 / 32.6
    #     top 5                            88.7 / 76.6 / 30.2
    #     top 20                           85.7 / 73.2 / 25.6
    #     every anchor, both directions    83.5 / 71.4 / 25.6
    #   budget 6000, no neighbour          91.3 / 82.3 / 53.5
    #   budget 6000, top 20                91.8 / 82.3 / 46.5
    #
    # Nothing moves multi-hop completeness -- the entire point of it -- and
    # doubling the budget does not rescue it: the slot spent on a neighbour
    # is worth more spent on the next scored turn.
    #
    # The diagnostic that made it look promising was a BASE-RATE error, and
    # it is worth leaving written down. 90.9 % of missing evidence turns are
    # adjacent to a packed one -- but 45 packed turns have ~90 neighbours,
    # about one of which is the missing turn. P(adjacent | missing) is high;
    # P(missing | adjacent) is ~1 %. Carrying them all buys one hit for
    # forty-five slots.
    #
    # Nothing replaces either of them, on purpose.

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

    # What the caller's prompt carries BESIDES the memory itself: the
    # instruction paragraphs and the delimiters, each present only when
    # something pulls it in. Zero for an empty packet, which renders as "".
    overhead_tokens = 0
    if packet_facts or packet_episodes:
        overhead_tokens = estimate_prose_tokens(cost_module.HEADER) + estimate_prose_tokens(
            cost_module.WRAPPER
        )
        if packet_episodes:
            overhead_tokens += estimate_prose_tokens(cost_module.EPISODES_HEADER)
        if contested_charged:
            overhead_tokens += estimate_prose_tokens(cost_module.CONTESTED_INSTRUCTIONS)

    packet = {
        "overhead_tokens": overhead_tokens,
        "facts": packet_facts,
        "episodes": packet_episodes,
        "warnings": warnings,
        "status": status,
        "empty_reason": empty_reason,
    }

    trace = ContextTrace(
        id=uuid.uuid4(),  # explicit: the column default only materializes on
        # flush, and Perf-2 skips the flush -- without this, trace.id below
        # would be None. Same semantics as the default, just earlier.
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
    # Perf-2: no `flush()` here. `trace.id` is a client-side uuid4 (see
    # app/models/trace.py), so it is known before any roundtrip, and both
    # callers commit right after this returns (routes/context.py,
    # routes/gateway.py) -- the flush was a redundant roundtrip carrying
    # the ~27 KB packet+decisions JSONB for nothing. A full write-behind
    # (respond first, persist after) is deliberately NOT done: the trace is
    # the proof the product sells, and a background write trades silent
    # trace loss plus inspect-races for milliseconds. Durability first.
    session.add(trace)
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
