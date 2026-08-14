"""Context window (mechanism F2, 15 aout Sprint 1): a packed slot never
stands alone.

- An episode packed by score carries its immediate temporal neighbor
  (radius 1 -- one event right before, one right after, same scope).
- A packed fact additionally carries the episode it was actually
  extracted from (its "source turn", via source_event_ids).

Isolation technique: give the neighbor/source event NO embedding so it
never enters the scored `episode_rows` pool in the first place (the
existing `Event.embedding.is_not(None)` filter excludes it there) --
the unified pool is greedy and would otherwise pack it directly by its
own (near-zero but nonzero) score before F2 ever runs, making the two
mechanisms impossible to tell apart. F2's own neighbor query does not
filter on embedding, so it still finds it.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select

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


async def _insert_unembedded_neighbor(subject_id: str, content: str, occurred_at) -> uuid.UUID:
    """Insert an Event directly, bypassing capture/consolidation, so it
    keeps `embedding=None` -- never a candidate in the scored episode
    pool, only findable through F2's own neighbor query."""
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
        d.get("episode_id") == str(neighbor_id) and d["reason_code"] == "episode_neighbor"
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
        # Strip its embedding after the fact: it must not be reachable
        # through the scored episode pool, only through F2's neighbor
        # query seeded by the fact's source_event_ids.
        source_event.embedding = None
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
        d.get("episode_id") == str(source_id) and d["reason_code"] == "fact_source_turn"
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
            # Just enough for the target episode itself (~32 tokens),
            # nothing left over for the neighbor.
            "budget_tokens": 35,
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
    assert any(
        d.get("episode_id") == str(neighbor_id) and d["reason_code"] == "over_budget"
        for d in decisions
    )
