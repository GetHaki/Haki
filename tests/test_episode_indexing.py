"""Episode indexing (mechanisms E1a/E3, 15 aout Sprint 1):

- E1a: episodes get the same full-text axis facts already had
  (`events.index_text` -> `events.search_vector`, migration 0022) --
  before this, an episode could only ever be found by embedding
  similarity, never by an exact lexical/keyword match.
- E3: true key merging -- an episode's own extracted facts are folded
  into its `index_text` at write time (app.consolidator), not merged with
  facts only at read time (the pre-existing "unified pool", 13 aout, still
  in place and untouched by this). An episode becomes findable through a
  term that appears ONLY in a fact extracted from it, never in its own
  raw payload.
"""

import uuid

from sqlalchemy import select, text

from app.consolidator import run_pending_consolidations
from app.db import async_session
from app.models import Event
from app.providers.fake import FakeProvider, mock_fact

ORG = "org_acme"
PROJECT = "prj_support"


async def _capture(client, events: list[dict]) -> None:
    response = await client.post(
        "/v1/capture",
        json={"idempotency_key": f"batch-{uuid.uuid4()}", "events": events},
    )
    assert response.status_code == 202


async def _run_worker() -> None:
    async with async_session() as session:
        await run_pending_consolidations(
            session, extractor=FakeProvider(), embedder=FakeProvider()
        )
        await session.commit()


def _event(subject_id: str, content: str, occurred_at: str, mock_facts: list[dict] | None = None) -> dict:
    payload: dict = {"messages": [{"role": "user", "content": content}]}
    if mock_facts is not None:
        payload["mock_facts"] = mock_facts
    return {
        "org_id": ORG,
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_id": subject_id,
        "kind": "chat_session",
        "occurred_at": occurred_at,
        "payload": payload,
    }


async def _tie_embeddings_and_recency(event_ids: list[uuid.UUID]) -> None:
    """Force two episodes to score identically on similarity and recency
    (same stored embedding, same occurred_at) so any remaining difference
    in which one gets packed can only come from the full-text axis this
    module tests -- same isolation technique as test_context.py's
    _collide_embedding for facts."""
    async with async_session() as session:
        rows = [await session.get(Event, event_id) for event_id in event_ids]
        shared_embedding = rows[0].embedding
        shared_occurred_at = rows[0].occurred_at
        for row in rows[1:]:
            row.embedding = shared_embedding
            row.occurred_at = shared_occurred_at
        await session.commit()


async def test_episode_found_by_exact_keyword_over_a_similarity_tied_rival(client):
    """E1a: with similarity and recency forced to tie exactly between two
    episodes, only the one whose text lexically matches the query has a
    non-zero full-text score -- a budget tight enough for exactly one
    episode must pack that one, never the other."""
    subject = "usr_episode_fts_1"
    await _capture(
        client,
        [
            _event(subject, "The user talked about kayaking on the river.", "2023-05-07T10:00:00Z"),
            _event(subject, "Zolgorvex mentioned a favorite pastime today.", "2023-05-07T11:00:00Z"),
        ],
    )
    await _run_worker()

    async with async_session() as session:
        events = (
            (await session.execute(select(Event).where(Event.subject_id == subject)))
            .scalars()
            .all()
        )
    await _tie_embeddings_and_recency([e.id for e in events])

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "Zolgorvex",
            "budget_tokens": 40,
        },
    )
    assert response.status_code == 200
    episodes = response.json()["packet"]["episodes"]
    assert len(episodes) == 1
    assert "Zolgorvex" in episodes[0]["excerpt"]


async def test_episode_index_text_folds_in_its_own_extracted_facts(client):
    """E3 (true key merging): the fact extracted from an event is
    concatenated into that SAME event's index_text at write time -- proven
    structurally (search_vector matches a term that exists ONLY in the
    fact, never in the raw payload) and behaviorally (the episode is
    served for a query on that term)."""
    subject = "usr_episode_fts_2"
    await _capture(
        client,
        [
            _event(
                subject,
                "Quick check-in, nothing eventful.",
                "2023-05-07T10:00:00Z",
                mock_facts=[
                    mock_fact("secret_code", {"code": "Bandersnatch42"}, subject_id=subject)
                ],
            )
        ],
    )
    await _run_worker()

    async with async_session() as session:
        event = (
            (await session.execute(select(Event).where(Event.subject_id == subject)))
            .scalars()
            .one()
        )
        assert "Bandersnatch42" in event.index_text
        assert "Bandersnatch42" not in event.payload["messages"][0]["content"]
        matched = (
            await session.execute(
                select(Event.id).where(
                    Event.id == event.id,
                    text(
                        "search_vector @@ websearch_to_tsquery('simple', 'Bandersnatch42')"
                    ),
                )
            )
        ).scalar_one_or_none()
        assert matched == event.id

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "Bandersnatch42",
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    episodes = response.json()["packet"]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["event_id"] == str(event.id)
