"""Provenance down to the turn: evidence_span, and what it unlocks.

The write gate (M1) has required a verbatim quote for every create/
supersede since it landed -- a candidate that cannot produce one must be
rejected with reason "no_evidence_span". The consolidator asked for it,
validated it, and threw it away.

These tests pin the three things persisting it buys: the quote itself,
the exact fact-to-turn link, and key merging that enriches ONE turn
instead of all of them.
"""

import uuid

from sqlalchemy import select

from app.db import async_session
from app.models import EpisodeChunk, Fact
from app.providers.fake import mock_fact
from tests.test_consolidator import run_worker

SESSION_TURNS = [
    {"role": "user", "content": "We talked about the weather for a while."},
    {"role": "user", "content": "I finally bought the Zolgorvex kayak this weekend."},
    {"role": "user", "content": "Then we moved on to something else entirely."},
]


async def _capture_session(client, subject: str, facts: list[dict]) -> None:
    payload = {"messages": [dict(turn) for turn in SESSION_TURNS]}
    response = await client.post(
        "/v1/capture",
        json={
            "idempotency_key": f"batch-{uuid.uuid4()}",
            "events": [
                {
                    "org_id": "org_acme",
                    "project_id": "prj_support",
                    "subject_type": "user",
                    "subject_id": subject,
                    "kind": "chat_session",
                    "occurred_at": "2026-05-01T10:00:00Z",
                    "payload": {**payload, "mock_facts": facts},
                }
            ],
        },
    )
    assert response.status_code == 202
    await run_worker()


async def _facts_of(subject: str) -> list[Fact]:
    async with async_session() as session:
        return list(
            (await session.execute(select(Fact).where(Fact.subject_id == subject)))
            .scalars()
            .all()
        )


async def _chunks_of(subject: str) -> list[EpisodeChunk]:
    async with async_session() as session:
        return list(
            (
                await session.execute(
                    select(EpisodeChunk)
                    .where(EpisodeChunk.subject_id == subject)
                    .order_by(EpisodeChunk.ordinal)
                )
            )
            .scalars()
            .all()
        )


async def test_the_evidence_span_is_persisted(client):
    subject = "usr_prov_1"
    await _capture_session(
        client,
        subject,
        [
            mock_fact(
                "owns_kayak",
                {"brand": "Zolgorvex"},
                subject_id=subject,
                evidence_span="I finally bought the Zolgorvex kayak this weekend.",
            )
        ],
    )
    [fact] = await _facts_of(subject)
    assert fact.evidence_span == "I finally bought the Zolgorvex kayak this weekend."


async def test_the_span_resolves_to_the_exact_turn_it_came_from(client):
    subject = "usr_prov_2"
    await _capture_session(
        client,
        subject,
        [
            mock_fact(
                "owns_kayak",
                {"brand": "Zolgorvex"},
                subject_id=subject,
                evidence_span="I finally bought the Zolgorvex kayak this weekend.",
            )
        ],
    )
    [fact] = await _facts_of(subject)
    chunks = await _chunks_of(subject)
    by_id = {chunk.id: chunk for chunk in chunks}
    assert fact.source_chunk_id is not None
    assert by_id[fact.source_chunk_id].ordinal == 1


async def test_a_span_that_matches_nothing_attributes_nothing(client):
    """Silence beats a wrong answer.

    Accepting the closest fuzzy candidate would attribute the fact to a
    turn it did not come from and pollute that turn's index. NULL keeps
    every consumer on the fallback it used before the link existed.
    """
    subject = "usr_prov_3"
    await _capture_session(
        client,
        subject,
        [
            mock_fact(
                "owns_kayak",
                {"brand": "Zolgorvex"},
                subject_id=subject,
                evidence_span="a sentence that appears in none of the turns",
            )
        ],
    )
    [fact] = await _facts_of(subject)
    assert fact.source_chunk_id is None


async def test_a_single_chunk_event_needs_no_span_to_be_attributed(client):
    """With one turn there is nothing to disambiguate.

    This is most of the product's traffic (`conversation.message`), and
    treating it as unknown would leave key merging switched off for it.
    """
    subject = "usr_prov_4"
    response = await client.post(
        "/v1/capture",
        json={
            "idempotency_key": f"batch-{uuid.uuid4()}",
            "events": [
                {
                    "org_id": "org_acme",
                    "project_id": "prj_support",
                    "subject_type": "user",
                    "subject_id": subject,
                    "kind": "conversation.message",
                    "occurred_at": "2026-05-01T10:00:00Z",
                    "payload": {
                        "role": "user",
                        "content": "nothing quotable here",
                        "mock_facts": [
                            mock_fact("mood", {"state": "calm"}, subject_id=subject)
                        ],
                    },
                }
            ],
        },
    )
    assert response.status_code == 202
    await run_worker()
    [fact] = await _facts_of(subject)
    assert fact.source_chunk_id is not None


async def test_key_merging_enriches_only_the_turn_the_fact_came_from(client):
    """K = V + fact, at the granularity that makes it worth doing.

    Before this mechanism the fact was folded into the index of the WHOLE
    event, so a fact from turn 2 of a twenty-turn session was appended to
    all twenty -- every turn matching a term none of them contains. Here
    the term exists only in the fact, and must appear in exactly one
    chunk's index_text while leaving the chunks' served text untouched.
    """
    subject = "usr_prov_5"
    await _capture_session(
        client,
        subject,
        [
            mock_fact(
                "secret_code",
                {"code": "Bandersnatch42"},
                subject_id=subject,
                evidence_span="I finally bought the Zolgorvex kayak this weekend.",
            )
        ],
    )
    chunks = await _chunks_of(subject)
    enriched = [chunk for chunk in chunks if "Bandersnatch42" in chunk.index_text]
    assert len(enriched) == 1, "the fact leaked into more than one turn's index"
    assert enriched[0].ordinal == 1
    # The served text is never touched: a fact must not be able to appear
    # inside what the agent reads as a verbatim quote.
    assert all("Bandersnatch42" not in chunk.text for chunk in chunks)


async def test_the_enriched_turn_is_retrievable_by_a_term_only_the_fact_holds(client):
    subject = "usr_prov_6"
    await _capture_session(
        client,
        subject,
        [
            mock_fact(
                "secret_code",
                {"code": "Bandersnatch42"},
                subject_id=subject,
                evidence_span="I finally bought the Zolgorvex kayak this weekend.",
            )
        ],
    )
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": subject,
            "query": "Bandersnatch42",
            "budget_tokens": 200,
        },
    )
    assert response.status_code == 200
    excerpts = " ".join(e["excerpt"] for e in response.json()["packet"]["episodes"])
    assert "Zolgorvex kayak" in excerpts, (
        "the turn enriched with the fact was not retrieved by the fact's own term"
    )
