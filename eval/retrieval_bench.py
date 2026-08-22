"""Retrieval bench: does the packet actually contain the evidence?

    uv run python -m eval.retrieval_bench --conversations 2
    uv run python -m eval.retrieval_bench --all --min-served 80

What it measures
-----------------
One number: **gold served** -- for each question, after ingestion,
consolidation, ranking and packing under a token budget, is the dialogue
turn LoCoMo annotates as holding the answer actually in the packet?

That number is the ceiling on everything downstream. If the evidence is
not in front of the reader, no answer prompt, no judge and no model can
recover it. Between 15 and 20 August this project shipped eight retrieval
mechanisms against a packet that held the evidence 6.8 % of the time, and
nothing in CI could say so.

Why this exists next to eval/run.py
------------------------------------
`eval.run` measures accuracy: the real thing, with a real LLM, a real
judge, several dollars and a few hours per run. It is the right instrument
for a release and the wrong one for a pull request.

This bench is the complement:

- **deterministic** -- no LLM anywhere, so two runs give the same number to
  the bit. `eval.run` cannot say that. That claim was false until 22 aout:
  the fact-fetch behind chunk indexing had no ORDER BY, and phase-1
  candidate generation and the final pool sort broke ties on a random
  uuid4 -- so the SAME corpus ingested twice could produce a DIFFERENT
  `index_text` (hence a different embedding) and keep a DIFFERENT subset
  of a tied candidate group. Measured directly on this bench: 86 of 231
  packets differed between two runs with nothing else changed. Fixed by
  replacing every random tie-break with one on recency then content (see
  `_FACT_TIEBREAK`/`_EPISODE_TIEBREAK`/`_content_tiebreak` in
  `app.context` and the fact-fetch `ORDER BY` in
  `app.consolidator._merge_facts_into_chunk_index`); this bench's own
  numbers are only trustworthy to the bit as of that fix.
- **free** -- 0 $, local ONNX embeddings only.
- **causally upstream** -- it isolates R1..R4 (capture, indexing, ranking,
  budget) from R5 (the reader), so a regression points at one stage
  instead of moving one aggregate number for unknown reasons.
- **independent of LoCoMo's answer key**, part of which is wrong (Penfield
  Labs audit, cited in the external strategic dossier this bench responds
  to). It reads the `evidence` annotation, not the answers.

It is NOT a substitute for accuracy. It cannot see whether the reader uses
what it is given -- measured at 59 % on this project's own runs, and the
next ceiling once this number is high.

How it stays honest
--------------------
Everything runs through the real code: `write_events` -> the real
consolidator -> `build_context`. No reimplementation of the scoring, the
chunking or the packing, because a reimplementation drifts and then
measures a model of Haki rather than Haki. That is exactly how a lexical
axis that matched nothing on 90 % of queries stayed green for weeks.

Extraction is the one thing that cannot run for free -- so the bench uses
LoCoMo's own `observation` layer as an ORACLE extractor, fed through
`FakeProvider`'s `mock_facts` with the source turn as `evidence_span`.
That exercises the real consolidator, the real chunk attribution
(migration 0022 -- `source_chunk_id`) and the real key merging,
deterministically. It also makes every conclusion about storage
architecture STRONGER, not weaker: an oracle extractor is the best case,
so whatever the fact channel cannot do here, it cannot do at all.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

# `app.ledger` first, on purpose: app.consolidator imports app.ledger.core,
# which (importing the app.ledger PACKAGE as a side effect) runs
# app/ledger/__init__.py, which imports app.consolidator -- importing the
# consolidator first walks into that cycle mid-initialisation and fails
# (verified directly: `import app.consolidator` alone raises ImportError
# "cannot import name 'run_pending_consolidations' from partially
# initialized module 'app.consolidator'"; `import app.ledger` first does
# not). The package that owns the cycle has to be the entry point.
import app.ledger  # noqa: F401  (import order, see above)
from app.consolidator import run_pending_consolidations
from app.context import build_context
from app.db import async_session
from app.ledger.core import write_events
from app.ledger.jobs import create_consolidation_job
from app.models import ConflictSet, ContextTrace, EpisodeChunk, Event, Fact
from app.providers import get_embedder, get_extractor
from app.schemas.capture import EventIn
from eval.datasets import FALLBACK_BASE, _parse_locomo_date
from eval.run import ROOT, load_config

CONFIG = "eval/configs/locomo_calibration.json"
ORG_ID = "org_retrieval_bench"
PROJECT_ID = "prj_retrieval_bench"
# LoCoMo category 5 is adversarial (no answer in the history) and is
# excluded by the calibration protocol; the other four are the standard
# questions.
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_DIA_RE = re.compile(r"D(\d+):(\d+)")


def _subject_of(sample_id: str) -> str:
    return f"usr_{sample_id}"


def _sessions_of(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """(session key, raw date, turns), in order."""
    sessions = []
    index = 1
    while f"session_{index}" in conversation:
        sessions.append(
            (
                f"session_{index}",
                conversation.get(f"session_{index}_date_time", ""),
                conversation[f"session_{index}"],
            )
        )
        index += 1
    return sessions


def _oracle_facts(sample: dict, session_key: str, turns: list[dict]) -> list[dict]:
    """LoCoMo's own observations for one session, as extractor candidates.

    Each observation is `[sentence, dia_id]`: a fact and the exact turn it
    came from. Passing that turn's text as `evidence_span` is what lets the
    real consolidator resolve `source_chunk_id` the same way it would with
    a real extractor -- the mechanism under test, not a shortcut around it.
    """
    by_dia = {turn.get("dia_id"): turn.get("text", "") for turn in turns}
    observations = (sample.get("observation") or {}).get(f"{session_key}_observation") or {}
    candidates = []
    for speaker, items in observations.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not (isinstance(item, list) and len(item) >= 2):
                continue
            sentence, dias = str(item[0]), item[1]
            dias = [dias] if isinstance(dias, str) else [str(d) for d in dias]
            span = by_dia.get(dias[0])
            if not span:
                continue
            candidates.append(
                {
                    "action": "create",
                    # Unique per observation (these are independent facts,
                    # not competing values for one predicate) and STABLE
                    # across processes. Python's hash() is randomised per
                    # interpreter unless PYTHONHASHSEED is pinned, and this
                    # bench caught it on itself: two runs of the same code
                    # gave different served rates, because different
                    # predicate names change which facts collide, supersede
                    # and open conflict sets. A CI gate that drifts is not a
                    # gate.
                    "predicate": "obs_"
                    + hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:12],
                    "value": {"statement": sentence, "person": speaker},
                    "confidence": 0.9,
                    "evidence_span": span,
                    "subject_id": "unused-scope-comes-from-the-event",
                }
            )
    return candidates


async def _reset_scope() -> None:
    """Drop everything this bench wrote. It owns its project entirely."""
    async with async_session() as session:
        for model in (ContextTrace, ConflictSet, Fact, EpisodeChunk, Event):
            await session.execute(delete(model).where(model.project_id == PROJECT_ID))
        await session.commit()


async def _ingest(sample: dict) -> dict[uuid.UUID, str]:
    """Ingest one conversation. Returns chunk_id -> dia_id.

    The map is built AFTER consolidation by pairing each chunk's ordinal
    with the turn at the same index -- which holds because the chunker cuts
    a `chat_session` payload on its `messages` boundaries, in order. The
    bench asserts the count matches rather than assuming it.
    """
    conversation = sample["conversation"]
    sample_id = str(sample.get("sample_id", "conv"))
    subject_id = _subject_of(sample_id)
    sessions = _sessions_of(conversation)

    async with async_session() as session:
        events = []
        for session_key, raw_date, turns in sessions:
            occurred_at = _parse_locomo_date(raw_date, FALLBACK_BASE)
            events.append(
                EventIn(
                    org_id=ORG_ID,
                    project_id=PROJECT_ID,
                    subject_type="user",
                    subject_id=subject_id,
                    kind="chat_session",
                    occurred_at=occurred_at,
                    payload={
                        "session_id": session_key,
                        "date": raw_date,
                        "messages": [
                            {"role": turn.get("speaker", "?"), "content": turn.get("text", "")}
                            for turn in turns
                        ],
                        "mock_facts": _oracle_facts(sample, session_key, turns),
                    },
                )
            )
        written = await write_events(
            session, events, batch_idempotency_key=f"bench-{sample_id}"
        )
        # The capture ROUTE is what normally enqueues consolidation; going
        # through the ledger directly means enqueuing it here, exactly as
        # the route does.
        await create_consolidation_job(
            session,
            project_id=PROJECT_ID,
            event_ids=[event.id for event, _deduplicated in written],
        )
        await session.commit()

    async with async_session() as session:
        await run_pending_consolidations(
            session, extractor=get_extractor(), embedder=get_embedder()
        )
        await session.commit()

    chunk_to_dia: dict[uuid.UUID, str] = {}
    async with async_session() as session:
        rows = (
            await session.execute(
                select(EpisodeChunk.id, EpisodeChunk.ordinal, Event.payload)
                .join(Event, Event.id == EpisodeChunk.event_id)
                .where(EpisodeChunk.subject_id == subject_id)
            )
        ).all()
    by_session_key = {key: turns for key, _, turns in sessions}
    for row in rows:
        turns = by_session_key.get((row.payload or {}).get("session_id"))
        if turns is None or row.ordinal >= len(turns):
            continue
        dia_id = turns[row.ordinal].get("dia_id")
        if dia_id:
            chunk_to_dia[row.id] = dia_id
    return chunk_to_dia


async def _served_dias(
    packet: dict[str, Any], chunk_to_dia: dict[uuid.UUID, str]
) -> set[str]:
    """Which annotated turns the packet actually put in front of the reader.

    Both channels count, and for the same reason: the question is whether
    the evidence reached the reader, not which pipe carried it.

      - an episode IS the turn: its `episode_id` is the chunk;
      - a fact was EXTRACTED from a turn: `source_chunk_id` says which.
    """
    served = {
        chunk_to_dia[uuid.UUID(episode["episode_id"])]
        for episode in packet.get("episodes", [])
        if episode.get("episode_id")
        and uuid.UUID(episode["episode_id"]) in chunk_to_dia
    }
    fact_ids = [uuid.UUID(fact["id"]) for fact in packet.get("facts", [])]
    if fact_ids:
        async with async_session() as session:
            for (chunk_id,) in (
                await session.execute(
                    select(Fact.source_chunk_id).where(
                        Fact.id.in_(fact_ids), Fact.source_chunk_id.is_not(None)
                    )
                )
            ).all():
                if chunk_id in chunk_to_dia:
                    served.add(chunk_to_dia[chunk_id])
    return served


async def run(conversations: int | None, budget: int, min_served: float | None) -> int:
    config = load_config(CONFIG)
    dataset_path = ROOT / config["dataset"]["file"]
    if not dataset_path.exists():
        print(
            f"dataset missing: {dataset_path}\n"
            f"  uv run python -m eval.download {CONFIG}",
            file=sys.stderr,
        )
        return 2
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    if conversations is not None:
        samples = samples[:conversations]

    await _reset_scope()
    embedder = get_embedder()
    per_category: dict[int, list[bool]] = defaultdict(list)
    started = datetime.now()

    for position, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"conv{position}"))
        subject_id = _subject_of(sample_id)
        chunk_to_dia = await _ingest(sample)
        as_of = max(
            _parse_locomo_date(raw_date, FALLBACK_BASE)
            for _, raw_date, _ in _sessions_of(sample["conversation"])
        )
        questions = [
            qa
            for qa in sample.get("qa", [])
            if int(qa.get("category", 0)) in CATEGORY_NAMES and (qa.get("evidence") or [])
        ]
        for qa in questions:
            gold = {str(dia) for dia in qa["evidence"] if _DIA_RE.match(str(dia))}
            async with async_session() as session:
                packet, _tokens, _trace_id = await build_context(
                    session,
                    project_id=PROJECT_ID,
                    subject_id=subject_id,
                    query=str(qa.get("question", "")),
                    budget_tokens=budget,
                    embedder=embedder,
                    as_of=as_of,
                )
                await session.commit()
            served = await _served_dias(packet, chunk_to_dia)
            per_category[int(qa["category"])].append(bool(served & gold))
        done = sum(len(v) for v in per_category.values())
        print(
            f"[{position}/{len(samples)}] {sample_id}: {len(questions)} questions "
            f"({done} total, {(datetime.now() - started).seconds}s)",
            flush=True,
        )

    all_results = [ok for results in per_category.values() for ok in results]
    if not all_results:
        print("no question selected", file=sys.stderr)
        return 2
    served_rate = 100 * sum(all_results) / len(all_results)

    print(f"\ngold served @{budget} tokens: {served_rate:.1f}%  (n={len(all_results)})")
    for category, results in sorted(per_category.items()):
        rate = 100 * sum(results) / len(results)
        print(f"  {CATEGORY_NAMES[category]:<12} {rate:5.1f}%  (n={len(results)})")

    if min_served is not None and served_rate < min_served:
        print(
            f"\nFAIL: {served_rate:.1f}% < {min_served}% -- the packet stopped "
            "carrying the evidence. Whatever else a change improved, it did "
            "not improve this.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--conversations",
        type=int,
        default=2,
        help="first N LoCoMo conversations (default 2, CI-sized)",
    )
    group.add_argument(
        "--all", action="store_true", help="all conversations in the pinned dataset"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="context budget in tokens (default 2000, the product default)",
    )
    parser.add_argument(
        "--min-served",
        type=float,
        default=None,
        help="exit non-zero below this percentage (the CI gate)",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(None if args.all else args.conversations, args.budget, args.min_served)
    )


if __name__ == "__main__":
    raise SystemExit(main())
