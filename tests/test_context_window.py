"""Context window (mechanism F2, 15 aout Sprint 1): a packed slot never
stands alone.

- An episode packed by score carries its immediate temporal neighbor
  (radius 1 -- one chunk right before, one right after, same scope).
- A packed fact additionally carries the episode it was actually
  extracted from (its "source turn", via source_event_ids).

Isolation technique: give the neighbor/source CHUNK no embedding, so it
never enters the scored `episode_rows` pool in the first place (the
`EpisodeChunk.embedding.is_not(None)` filter excludes it there) -- the
unified pool is greedy and would otherwise pack it directly by its own
(near-zero but nonzero) score before F2 ever runs, making the two
mechanisms impossible to tell apart. F2's own neighbor query does not
filter on embedding, so it still finds it.

Since 21 Aug (migration 0024) the retrievable unit is an `episode_chunks`
row rather than a whole event, so the isolation moves with it: the
fixture inserts the event AND its chunk, and it is the chunk's embedding
that is left NULL. The intent of these tests is unchanged.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from app.consolidator import run_pending_consolidations
from app.db import async_session
from app.models import EpisodeChunk, Event
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


async def _insert_unembedded_neighbor(subject_id: str, content: str, occurred_at) -> uuid.UUID:
    """Insert an Event and its chunk directly, bypassing consolidation.

    The chunk keeps `embedding=None`, so it is never a candidate in the
    scored episode pool and is only reachable through F2's own neighbor
    query -- which is what makes these tests able to tell the two
    mechanisms apart.
    """
    event = Event(
        org_id=ORG,
        project_id=PROJECT,
        subject_type="user",
        subject_id=subject_id,
        kind="chat_session",
        occurred_at=occurred_at,
        payload={"messages": [{"role": "user", "content": content}]},
        hash=f"sha256:{uuid.uuid4().hex}",
        idempotency_key=f"neighbor-{uuid.uuid4()}",
    )
    async with async_session() as session:
        session.add(event)
        await session.flush()
        session.add(
            EpisodeChunk(
                event_id=event.id,
                ordinal=0,
                project_id=PROJECT,
                subject_id=subject_id,
                occurred_at=event.occurred_at,
                origin_trust=event.origin_trust,
                text=f"user: {content}",
                index_text=f"user: {content}",
                embedding=None,
            )
        )
        await session.commit()
        return event.id


async def test_packed_episode_carries_its_unembedded_temporal_neighbor(client):
    subject = "usr_ctxwindow_1"
    target_at = "2023-05-07T12:00:00Z"
    await _capture(
        client,
        [_event(subject, "Zolgorvex mentioned a favorite pastime today.", target_at)],
    )
    await _run_worker()

    async with async_session() as session:
        target = (
            (await session.execute(select(Event).where(Event.subject_id == subject)))
            .scalars()
            .one()
        )
    after_at = datetime(2023, 5, 7, 13, 0, tzinfo=timezone.utc)
    neighbor_id = await _insert_unembedded_neighbor(subject, "Unrelated later chat.", after_at)

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "Zolgorvex",
            "budget_tokens": 200,
        },
    )
    assert response.status_code == 200
    body = response.json()
    episodes = body["packet"]["episodes"]
    by_id = {e["event_id"]: e for e in episodes}
    assert set(by_id) == {str(target.id), str(neighbor_id)}
    assert by_id[str(target.id)]["context_neighbor"] is False
    assert by_id[str(neighbor_id)]["context_neighbor"] is True

    inspect = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": PROJECT, "subject_id": subject},
    )
    decisions = inspect.json()["decisions"]
    assert any(
        # The trace addresses the ranked unit (the chunk); the packet
        # carries both ids, so the correlation goes through episode_id.
        d.get("episode_id") == by_id[str(neighbor_id)]["episode_id"]
        and d["reason_code"] == "episode_neighbor"
        for d in decisions
    )


async def test_packed_fact_carries_its_unembedded_source_turn(client):
    subject = "usr_ctxwindow_2"
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
        source_event = (
            (await session.execute(select(Event).where(Event.subject_id == subject)))
            .scalars()
            .one()
        )
        # Strip the CHUNK embeddings after the fact: the source turn must
        # not be reachable through the scored episode pool, only through
        # F2's neighbor query seeded by the fact's source_event_ids.
        for chunk in (
            (
                await session.execute(
                    select(EpisodeChunk).where(EpisodeChunk.event_id == source_event.id)
                )
            )
            .scalars()
            .all()
        ):
            chunk.embedding = None
        await session.commit()
        source_id = source_event.id

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "Bandersnatch42",
            "budget_tokens": 200,
        },
    )
    assert response.status_code == 200
    body = response.json()
    packet = body["packet"]
    assert len(packet["facts"]) == 1
    episodes = packet["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["event_id"] == str(source_id)
    assert episodes[0]["context_neighbor"] is True

    inspect = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": PROJECT, "subject_id": subject},
    )
    decisions = inspect.json()["decisions"]
    assert any(
        d.get("episode_id") == episodes[0]["episode_id"]
        and d["reason_code"] == "fact_source_turn"
        for d in decisions
    )


async def test_context_window_respects_the_budget(client):
    """A neighbor that would push the packet over budget is excluded, not
    silently squeezed in -- same over_budget contract as everything else
    packed in this function."""
    subject = "usr_ctxwindow_3"
    target_at = "2023-05-07T12:00:00Z"
    await _capture(
        client,
        [_event(subject, "Zolgorvex mentioned a favorite pastime today.", target_at)],
    )
    await _run_worker()

    after_at = datetime(2023, 5, 7, 13, 0, tzinfo=timezone.utc)
    neighbor_id = await _insert_unembedded_neighbor(subject, "Unrelated later chat.", after_at)

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject,
            "query": "Zolgorvex",
            # Just enough for the target chunk itself: the date + kind
            # prefix and the turn text come to ~20 estimated tokens, the
            # neighbour chunk to ~14. Recalibrated on 21 Aug, when the
            # served unit became a turn instead of a whole event (~32).
            "budget_tokens": 25,
        },
    )
    assert response.status_code == 200
    body = response.json()
    episodes = body["packet"]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["context_neighbor"] is False
    assert str(neighbor_id) not in {e["event_id"] for e in episodes}

    inspect = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": PROJECT, "subject_id": subject},
    )
    decisions = inspect.json()["decisions"]
    # The neighbour is excluded, so it is not in the packet to read an id
    # from: its chunk id comes from the database instead.
    async with async_session() as session:
        neighbor_chunk_id = (
            await session.execute(
                select(EpisodeChunk.id).where(EpisodeChunk.event_id == neighbor_id)
            )
        ).scalar_one()
    assert any(
        d.get("episode_id") == str(neighbor_chunk_id)
        and d["reason_code"] == "over_budget"
        for d in decisions
    )
