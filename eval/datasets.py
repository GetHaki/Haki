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
  `category` 1..5 (1 single-hop, 2 temporal, 3 multi-hop, 4 open-domain,
  5 adversarial/abstention).

Both loaders return a uniform list of `Question`. Selection (`select`) is
deterministic: dataset order, optional type filter, then first N.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Fallback base date when a dataset date cannot be parsed: sessions keep
# their relative chronology (one hour apart) so bitemporal ingestion still
# sees updates in the right order.
FALLBACK_BASE = datetime(2023, 1, 1, tzinfo=timezone.utc)

LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
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
        questions.append(
            Question(
                qid=str(item["question_id"]),
                qtype=qtype,
                question=str(item["question"]),
                answer=str(answer),
                question_date=item.get("question_date"),
                abstention_expected=qtype in ABSTENTION_TYPES
                or str(item["question_id"]).endswith("_abs"),
                sessions=sessions,
                evidence_sessions=[s for s in sessions if s.session_id in evidence_ids],
                history_id=str(item["question_id"]),  # each question has its own haystack
            )
        )
    return questions


def load_locomo(path: str | Path) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions: list[Question] = []
    for conv in raw:
        sample_id = str(conv.get("sample_id", f"conv{len(questions)}"))
        conversation = conv["conversation"]
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
                )
            )
    return questions


LOADERS = {"longmemeval": load_longmemeval, "locomo": load_locomo}


def select(
    questions: list[Question],
    subset: int | None = None,
    types: list[str] | None = None,
) -> list[Question]:
    """Deterministic selection: dataset order, optional type filter, first N."""
    if types:
        wanted = {t.strip() for t in types if t.strip()}
        questions = [q for q in questions if q.qtype in wanted]
    if subset is not None:
        questions = questions[:subset]
    return questions


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
    prompt."""
    lines: list[str] = []
    for fact in facts:
        value = json.dumps(fact.get("value", {}), ensure_ascii=False, sort_keys=True)
        valid_from = fact.get("valid_from") or "?"
        lines.append(f"- {fact.get('predicate', '?')}: {value} (valid from {valid_from})")
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
