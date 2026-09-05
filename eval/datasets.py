"""Dataset loaders for the evaluation harness.

Two formats are supported:

- LongMemEval (https://github.com/xiaowu0162/LongMemEval, data on
  huggingface.co/datasets/xiaowu0162/longmemeval-cleaned): one JSON list of
  questions, each with `haystack_sessions` (list of sessions, each a list of
  {"role", "content"} messages), `haystack_dates` ("2023/05/30 (Tue) 23:40"),
  `question_type` (single-session-user/-assistant/-preference, multi-session,
  temporal-reasoning, knowledge-update, abstention).
- LoCoMo (https://github.com/snap-research/locomo, data/locomo10.json): one
  JSON list of conversations; each conversation has `session_N` /
  `session_N_date_time` ("1:56 pm on 8 May, 2023") keys and a `qa` list with
  `category` 1..5 (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
  5 adversarial/abstention).

Both loaders return a uniform list of `Question`. Selection (`select`) is
deterministic: optional type filter, then a proportional stratified sample
by question type (seeded, same sample in every process) -- see `select`'s
own docstring for why this replaced plain first-N on 21 Aug.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Fallback base date when a dataset date cannot be parsed: sessions keep
# their relative chronology (one hour apart) so bitemporal ingestion still
# sees updates in the right order.
FALLBACK_BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# 15 aout: corrected against the OFFICIAL LoCoMo eval script
# (snap-research/locomo, task_eval/evaluation.py) -- category 1 gets
# multi-hop-style partial-answer F1 splitting there, category 3 gets the
# `answer.split(';')[0]` pre-processing distinctive of open-domain/
# commonsense questions (13 questions/conversation, speculative framing --
# cross-checked directly against eval/data/locomo10.json). The previous
# mapping here (1: single-hop, 3: multi-hop, 4: open-domain) had these
# three swapped -- only 2 (temporal) and 5 (adversarial) were ever correct.
# This silently mislabeled every per-category breakdown in every LoCoMo
# run and diagnostic before this fix; overall accuracy figures are
# unaffected (they never depended on category naming), only the
# per-qtype attribution.
LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# Question types where the gold answer means "not answerable from history".
ABSTENTION_TYPES = {"abstention", "adversarial"}

CHARS_PER_TOKEN = 4  # documented rough estimate, used only for truncation


@dataclass
class Message:
    speaker: str  # dataset role ("user"/"assistant") or speaker name
    content: str


@dataclass
class Session:
    session_id: str
    date: datetime
    messages: list[Message] = field(default_factory=list)


@dataclass
class Question:
    qid: str
    qtype: str  # dataset type, normalized (see loaders)
    question: str
    answer: str
    question_date: str | None
    abstention_expected: bool
    sessions: list[Session]
    evidence_sessions: list[Session] = field(default_factory=list)
    # Identifies the ingested history: several questions may share one
    # history (LoCoMo: same conversation) -> ingest once, like real usage.
    history_id: str = ""
    # 14 aout, mecanisme D (research/Diagnostic_Couverture_2026-08-14.md):
    # the point in time to pass as `as_of` to POST /v1/context, so a
    # question dated years in the past does not get every volatile/
    # ephemeral fact judged stale against today's real wall clock.
    # LongMemEval: the dataset's own question_date, parsed. LoCoMo: no
    # per-question date exists in the source data, so the last ingested
    # session's date stands in for "now" from the conversation's own point
    # of view -- the natural reading of "the question is being asked right
    # after this conversation".
    as_of: datetime | None = None
    # Mechanism E4 (15 aout, Sprint 1): the real named speakers of this
    # history, in source order (LoCoMo: conversation["speaker_a"/"speaker_b"]
    # -- always exactly 2, both real names, never "user"/"assistant").
    # Empty for LongMemEval, whose "speaker" is already the normalized
    # "user"/"assistant" role -- the shared-store attribution problem this
    # mechanism targets does not exist there. Populated regardless of
    # whether the speaker-mirror protocol is enabled for this run; it is
    # only ACTED on when `config["speaker_mirror"]` is set (see eval/run.py).
    speakers: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _parse_longmemeval_date(raw: str, fallback: datetime) -> datetime:
    try:
        dt = datetime.strptime(raw.strip(), "%Y/%m/%d (%a) %H:%M")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return fallback


def _parse_locomo_date(raw: str, fallback: datetime) -> datetime:
    # "1:56 pm on 8 May, 2023"
    try:
        dt = datetime.strptime(raw.strip(), "%I:%M %p on %d %B, %Y")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return fallback


def load_longmemeval(path: str | Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions: list[Question] = []
    for item in raw:
        sessions: list[Session] = []
        for i, (ids, date_str, sess) in enumerate(
            zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
            )
        ):
            sid = "+".join(ids) if isinstance(ids, list) else str(ids)
            date = _parse_longmemeval_date(
                date_str, FALLBACK_BASE.replace(hour=0) if i == 0 else sessions[-1].date.replace(hour=i % 24)
            )
            sessions.append(
                Session(
                    session_id=sid,
                    date=date,
                    messages=[
                        Message(speaker=m.get("role", "?"), content=m.get("content", ""))
                        for m in sess
                    ],
                )
            )
        qtype = str(item.get("question_type", "unknown"))
        answer = item.get("answer", "")
        if isinstance(answer, list):
            answer = "; ".join(str(a) for a in answer)
        evidence_ids = set(item.get("answer_session_ids", []))
        raw_question_date = item.get("question_date")
        questions.append(
            Question(
                qid=str(item["question_id"]),
                qtype=qtype,
                question=str(item["question"]),
                answer=str(answer),
                question_date=raw_question_date,
                abstention_expected=qtype in ABSTENTION_TYPES
                or str(item["question_id"]).endswith("_abs"),
                sessions=sessions,
                evidence_sessions=[s for s in sessions if s.session_id in evidence_ids],
                history_id=str(item["question_id"]),  # each question has its own haystack
                as_of=_parse_longmemeval_date(
                    raw_question_date or "",
                    sessions[-1].date if sessions else FALLBACK_BASE,
                ),
            )
        )
    return questions


def load_locomo(path: str | Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions: list[Question] = []
    for conv in raw:
        sample_id = str(conv.get("sample_id", f"conv{len(questions)}"))
        conversation = conv["conversation"]
        # Mechanism E4: the two real speaker names, straight from the
        # source (not inferred from message text) -- present on every
        # LoCoMo conversation in the official dataset shape.
        speakers = [
            s for s in (conversation.get("speaker_a"), conversation.get("speaker_b")) if s
        ]
        sessions: list[Session] = []
        index = 1
        while f"session_{index}" in conversation:
            raw_date = conversation.get(f"session_{index}_date_time", "")
            fallback = (
                FALLBACK_BASE if not sessions else sessions[-1].date.replace(hour=index % 24)
            )
            sessions.append(
                Session(
                    session_id=f"{sample_id}_s{index}",
                    date=_parse_locomo_date(raw_date, fallback),
                    messages=[
                        Message(speaker=m.get("speaker", "?"), content=m.get("text", ""))
                        for m in conversation[f"session_{index}"]
                    ],
                )
            )
            index += 1
        for q_idx, qa in enumerate(conv.get("qa", [])):
            category = int(qa.get("category", 0))
            qtype = LOCOMO_CATEGORIES.get(category, f"category-{category}")
            evidence_sessions = sorted(
                {
                    int(m.group(1))
                    for dia in qa.get("evidence", []) or []
                    if (m := re.match(r"D(\d+):", str(dia)))
                }
            )
            answer = qa.get("answer", "")
            questions.append(
                Question(
                    qid=f"{sample_id}_q{q_idx}",
                    qtype=qtype,
                    question=str(qa["question"]),
                    answer="unanswerable" if answer is None else str(answer),
                    question_date=None,
                    abstention_expected=qtype in ABSTENTION_TYPES or answer is None,
                    sessions=sessions,
                    evidence_sessions=[
                        s for i, s in enumerate(sessions, start=1) if i in evidence_sessions
                    ],
                    history_id=sample_id,  # one conversation = one ingested history
                    # No per-question date in LoCoMo's source data -- the
                    # last session's date stands in for "now" from the
                    # conversation's own point of view (see the Question
                    # docstring above).
                    as_of=sessions[-1].date if sessions else None,
                    speakers=speakers,
                )
            )
    return questions


LOADERS = {"longmemeval": load_longmemeval, "locomo": load_locomo}


# Default sampling seed. Pinned rather than random: two runs of the same
# config must select the same questions, or their numbers are not
# comparable and a trajectory made of them means nothing.
DEFAULT_SEED = 42


def _stable_order_key(seed: int, qid: str) -> str:
    """A per-question shuffle key that is the same in every process.

    sha1, not Python's `hash()`: hash() is randomised per interpreter
    unless PYTHONHASHSEED is pinned, so it would reshuffle the sample on
    every run -- the exact bug this function exists to prevent (and one
    this project has already been bitten by, see eval/retrieval_bench.py).
    """
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def composition(questions: list[Question]) -> dict[str, int]:
    """How many questions of each type -- what a run must publish about its sample."""
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.qtype] = counts.get(question.qtype, 0) + 1
    return dict(sorted(counts.items()))


def select(
    questions: list[Question],
    subset: int | None = None,
    types: list[str] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> list[Question]:
    """Select `subset` questions, keeping the type mix of the full set.

    Why this is not "the first N"
    -----------------------------
    Until 21 Aug it was: `questions[:subset]`, i.e. dataset order. On
    LoCoMo, dataset order is conversation order -- so a 180-question run
    and a 458-question run did not sample the same conversations, did not
    sample the same question types, and were not comparable to each other
    or to a full run. This project published a trajectory (17.1 % ->
    30.6 % -> 31.4 %) built from exactly those three incomparable samples.

    Proportional stratified sampling by `qtype`, with the remainder
    apportioned largest-first so the totals add up exactly. Within a
    stratum the order is a stable sha1 shuffle keyed by `seed`, so the same
    (seed, subset) always yields the same questions, in every process, on
    every machine -- and a different seed gives an independent sample,
    which is what makes a variance estimate possible at all.

    The returned list is in DATASET order, not sample order: shards are cut
    from it by history_id (see `shard`), and ingestion cost depends on
    keeping a history's questions together.

    `stratify=False` restores the old first-N behaviour, for reproducing a
    pre-21-Aug number on purpose.
    """
    if types:
        wanted = {t.strip() for t in types if t.strip()}
        questions = [q for q in questions if q.qtype in wanted]
    if subset is None or subset >= len(questions):
        return questions
    if not stratify:
        return questions[:subset]

    by_type: dict[str, list[Question]] = {}
    for question in questions:
        by_type.setdefault(question.qtype, []).append(question)

    # Largest-remainder apportionment: floor every quota, then hand the
    # leftover places to the largest remainders. Guarantees the quotas sum
    # to `subset` exactly, and that no non-empty stratum is silently
    # dropped when its share rounds to zero.
    total = len(questions)
    exact = {qtype: len(group) * subset / total for qtype, group in by_type.items()}
    quotas = {qtype: int(value) for qtype, value in exact.items()}
    for qtype in sorted(
        exact, key=lambda t: (-(exact[t] - quotas[t]), t)
    )[: subset - sum(quotas.values())]:
        quotas[qtype] += 1

    chosen: set[str] = set()
    for qtype, group in by_type.items():
        ordered = sorted(group, key=lambda q: _stable_order_key(seed, q.qid))
        chosen.update(q.qid for q in ordered[: quotas[qtype]])
    return [q for q in questions if q.qid in chosen]


def shard(
    questions: list[Question],
    shard_index: int,
    shard_count: int,
) -> list[Question]:
    """Split `questions` into `shard_count` groups for parallel cloud runs
    (a single GitHub Actions job caps out at 6h, far short of a full
    LoCoMo/LongMemEval run), grouped by history_id (or qid when unset) so a
    shared ingested history is never split across two shards. Critical for
    LoCoMo: ~200 questions can share ONE conversation (eval/run.py's
    `ingested` cache is per-process, not shared across shards) — splitting
    them would make every shard that touches that conversation re-ingest it
    from scratch, multiplying real LLM extraction cost by however many
    shards happen to touch it. Round-robin assignment by first appearance of
    each history_id, deterministic given the same `questions` order in.
    """
    if shard_count <= 1:
        return questions
    order: list[str] = []
    seen: set[str] = set()
    for question in questions:
        key = question.history_id or question.qid
        if key not in seen:
            seen.add(key)
            order.append(key)
    assigned = {key: i % shard_count for i, key in enumerate(order)}
    return [
        question
        for question in questions
        if assigned[question.history_id or question.qid] == shard_index
    ]


def render_transcript(sessions: list[Session]) -> str:
    lines: list[str] = []
    for session in sessions:
        lines.append(f"--- session {session.session_id} ({session.date:%Y-%m-%d %H:%M} UTC) ---")
        for message in session.messages:
            lines.append(f"{message.speaker}: {message.content}")
    return "\n".join(lines)


def render_mem0_transcript(sessions: list[Session]) -> str:
    """15 aout, Sprint 0 calibration: mirrors mem0's own
    `RAGManager.clean_chat_history` (evaluation/src/rag.py) exactly --
    `f"{timestamp} | {speaker}: {text}\\n"` per message, no session-boundary
    markers (unlike `render_transcript` above, which is Haki's own baseline
    rendering and stays untouched for every non-calibration run).

    mem0's source dataset carries a timestamp per MESSAGE; Haki's Session
    model only carries one date per SESSION (see the `Session` dataclass),
    so every message in a session is stamped with that session's date --
    a documented approximation, not a divergence in structure (order and
    granularity of information shown to the model are unchanged, only the
    finest-grained per-message timestamp is unavailable in our own ingested
    data model)."""
    lines: list[str] = []
    for session in sessions:
        for message in session.messages:
            lines.append(f"{session.date.isoformat()} | {message.speaker}: {message.content}")
    return "\n".join(lines)


def render_lme_session_history(sessions: list[Session]) -> str:
    """15 aout, Sprint 0 calibration: mirrors the official LongMemEval
    full-context baseline's history rendering exactly
    (src/generation/run_generation.py, `retriever_type="orig-session"`,
    `history_format="json"`, `useronly=false`) -- one block per session:
    '\\n### Session {i}:\\nSession Date: {date}\\nSession Content:\\n{json}\\n',
    where the JSON is the session's raw [{"role", "content"}, ...] turns
    (`useronly=false` keeps both roles). Sessions are already chronological
    in Haki's own loaded data (see load_longmemeval), matching the source
    script's own `retrieved_chunks.sort(key=lambda x: x[0])`."""
    blocks: list[str] = []
    for i, session in enumerate(sessions, start=1):
        turns = [{"role": m.speaker, "content": m.content} for m in session.messages]
        sess_string = "\n" + json.dumps(turns, ensure_ascii=False)
        blocks.append(
            f"\n### Session {i}:\nSession Date: {session.date.isoformat()}\n"
            f"Session Content:\n{sess_string}\n"
        )
    return "".join(blocks)


def truncate_sessions(
    sessions: list[Session], max_tokens: int
) -> tuple[list[Session], bool]:
    """Keep the most recent whole sessions that fit the token budget
    (chars/4 estimate). Returns (kept sessions in chronological order,
    truncated flag)."""
    kept: list[Session] = []
    total = 0
    for session in reversed(sessions):
        cost = estimate_tokens(render_transcript([session]))
        if kept and total + cost > max_tokens:
            return list(reversed(kept)), True
        kept.append(session)
        total += cost
    return list(reversed(kept)), len(kept) < len(sessions)


def render_facts(facts: list[dict]) -> str:
    """Render a Haki ContextPacket's facts as the memory block of the answer
    prompt.

    Bench-2: beyond predicate + value + valid_from, the reader now sees the
    identity qualifiers (the condition the fact holds under -- team, person,
    ...), the end of the validity interval when one was set by a
    supersession, and whether the fact is one half of an open conflict. The
    answer prompt already instructs grouping "by what facts are actually
    about" and resolving by dates -- these fields are what that instruction
    was missing to act on. Absent fields render exactly as before, so old
    packets (and unity tests on this format) keep working unchanged.
    """
    lines: list[str] = []
    for fact in facts:
        value = json.dumps(fact.get("value", {}), ensure_ascii=False, sort_keys=True)
        valid_from = fact.get("valid_from") or "?"
        extras: list[str] = []
        qualifiers = fact.get("qualifiers") or {}
        if qualifiers:
            scope = json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)
            extras.append(f"holds when {scope}")
        valid_to = fact.get("valid_to")
        if valid_to:
            extras.append(f"valid until {valid_to}")
        if fact.get("contested"):
            extras.append("contested -- a conflicting value exists, do not trust either side alone")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        lines.append(
            f"- {fact.get('predicate', '?')}: {value} (valid from {valid_from}){suffix}"
        )
    return "\n".join(lines) if lines else "(no facts in memory)"


def render_episodes(episodes: list[dict]) -> str:
    """Render a Haki ContextPacket's episodes (dated source events)."""
    lines: list[str] = []
    for episode in episodes:
        occurred = episode.get("occurred_at") or "?"
        lines.append(f"- [{occurred}] {episode.get('kind', '?')}: {episode.get('excerpt', '')}")
    return "\n".join(lines) if lines else "(no dated events in memory)"


# Retrieval-need classification (12 aout, external feedback): what a
# question actually needs varies by shape. "What's the current value of
# X" is served well by a handful of facts; "why/how did X" or a
# hypothetical needs the source narrative; "how many X" is a count a
# retrieval packet (facts OR episodes) can't reliably answer at all, since
# neither guarantees every instance survived extraction or made the
# packet. This labels each question with its dominant need so failures
# become diagnosable by category instead of one undifferentiated bucket.
# Superseded the original motivation (informing a fixed facts/episodes
# budget split) once app/context/__init__.py moved to a unified ranked
# pool (13 aout, key merging) instead of any fixed share — this breakdown
# is still useful diagnostically, and would inform per-need SCORING
# weights if that's ever worth tuning, just not a budget split anymore.
RETRIEVAL_NEEDS = ("count", "narrative", "point_value")

_COUNT_RE = re.compile(r"^\s*how\s+(many|much|often)\b", re.IGNORECASE)
_NARRATIVE_RE = re.compile(
    r"^\s*(why|would|could|should|how (?:do|does|did)|in what way)\b"
    r"|\b(describe|compare|summarize|summarise|explain)\b",
    re.IGNORECASE,
)


def classify_retrieval_need(question: str) -> str:
    """Heuristic, not ML: keyword/prefix rules on the question text alone,
    order matters (count checked first so "how often" doesn't fall into
    the "how do/does/did" narrative branch). Defaults to "point_value" --
    the common case ("what/where/when/who is/was X") needs a specific
    current or dated fact, which is exactly what the fact ledger already
    targets."""
    if _COUNT_RE.search(question):
        return "count"
    if _NARRATIVE_RE.search(question):
        return "narrative"
    return "point_value"
