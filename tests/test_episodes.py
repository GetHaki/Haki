"""Episodic memory (sprint 10): the consolidator embeds source events and
/v1/context serves dated episodes alongside facts, under the same token
budget, scoped like facts. Also covers predicate-stable supersession
(sprint-10 extraction contract): reusing the exact predicate replaces the
fact instead of opening a conflict.
"""

import uuid

from sqlalchemy import select

from app.consolidator import run_pending_consolidations
from app.db import async_session
from app.models import ConflictSet, Event, Fact, FactStatus
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


def _dated_event(subject_id: str, content: str, occurred_at: str) -> dict:
    return {
        "org_id": ORG,
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_id": subject_id,
        "kind": "chat_session",
        "occurred_at": occurred_at,
        "payload": {"messages": [{"role": "user", "content": content}]},
    }


async def test_consolidation_embeds_events(client):
    await _capture(client, [_dated_event("usr_epi", "I went to the LGBTQ support group.", "2023-05-07T14:00:00Z")])
    await _run_worker()
    async with async_session() as session:
        event = (await session.execute(select(Event))).scalars().one()
    assert event.embedding is not None
    assert len(event.embedding) == 384


async def test_context_serves_dated_episode(client):
    await _capture(client, [_dated_event("usr_epi", "I went to the LGBTQ support group.", "2023-05-07T14:00:00Z")])
    await _run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": "usr_epi",
            "query": "when did the user go to the support group?",
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    packet = response.json()["packet"]
    assert len(packet["episodes"]) == 1
    episode = packet["episodes"][0]
    assert episode["occurred_at"].startswith("2023-05-07")
    assert "support group" in episode["excerpt"]
    assert episode["kind"] == "chat_session"

    # The decision trace records the episode inclusion.
    trace_id = response.json()["trace_id"]
    inspect = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": PROJECT, "subject_id": "usr_epi"},
    )
    decisions = inspect.json()["decisions"]
    assert any(
        d.get("episode_id") == episode["event_id"] and d["action"] == "included"
        for d in decisions
    )


async def test_episodes_respect_token_budget(client):
    await _capture(client, [_dated_event("usr_epi", "a long story " * 50, "2023-05-07T14:00:00Z")])
    await _run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": "usr_epi",
            "query": "the story",
            "budget_tokens": 10,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["packet"]["episodes"] == []
    assert body["token_count"] <= 10
    inspect = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": PROJECT, "subject_id": "usr_epi"},
    )
    assert any(
        d.get("episode_id") and d["reason_code"] == "over_budget"
        for d in inspect.json()["decisions"]
    )


async def test_episodes_are_scope_isolated(client):
    await _capture(client, [_dated_event("usr_a", "Alice adopted a cat.", "2023-05-07T14:00:00Z")])
    await _run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": "usr_b",
            "query": "did anyone adopt a cat?",
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    assert response.json()["packet"]["episodes"] == []


async def test_packet_without_events_keeps_episodes_empty(client):
    """Backward compatibility: packets always carry `episodes` (default [])."""
    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": "usr_nobody",
            "query": "anything",
            "budget_tokens": 900,
        },
    )
    assert response.status_code == 200
    assert response.json()["packet"] == {
        "facts": [],
        "episodes": [],
        "warnings": [
            "missing_purpose: 'purpose' is recommended on context calls (warning only in V1)"
        ],
        "status": "degraded",
    }


async def test_supersede_reuses_exact_predicate_no_conflict(client):
    """Sprint-10 extraction contract: an update reusing the EXACT predicate
    replaces the fact (supersession), it does NOT open a conflict set."""
    def event(mock_facts, occurred_at):
        return {
            "org_id": ORG,
            "project_id": PROJECT,
            "subject_type": "user",
            "subject_id": "usr_runner",
            "kind": "conversation.message",
            "occurred_at": occurred_at,
            "payload": {"role": "user", "content": "...", "mock_facts": mock_facts},
        }

    await _capture(client, [
        event([mock_fact("personal_best_5k", {"time": "27:12"}, subject_id="usr_runner")], "2023-05-23T10:00:00Z")
    ])
    await _run_worker()
    await _capture(client, [
        event(
            [mock_fact(
                "personal_best_5k", {"time": "25:50"},
                subject_id="usr_runner",
                action="supersede", supersedes_predicate="personal_best_5k",
            )],
            "2023-05-30T10:00:00Z",
        )
    ])
    await _run_worker()

    async with async_session() as session:
        facts = (
            (await session.execute(select(Fact).where(Fact.subject_id == "usr_runner")))
            .scalars().all()
        )
        conflicts = (
            (await session.execute(select(ConflictSet).where(ConflictSet.status == "open")))
            .scalars().all()
        )
    active = [f for f in facts if f.status is FactStatus.active]
    assert len(active) == 1
    assert active[0].predicate == "personal_best_5k"
    assert active[0].value == {"time": "25:50"}
    assert [f.status for f in facts].count(FactStatus.superseded) == 1
    assert conflicts == []


def test_sdk_prompt_context_renders_episodes():
    from haki.runtime import build_prompt_context

    prompt = build_prompt_context(
        {
            "facts": [],
            "episodes": [
                {
                    "event_id": "evt-1",
                    "kind": "chat_session",
                    "occurred_at": "2023-05-07T14:00:00+00:00",
                    "excerpt": "user: I went to the support group.",
                }
            ],
            "warnings": [],
        }
    )
    assert "Dated events from the source history" in prompt
    assert "2023-05-07" in prompt
    assert "support group" in prompt
    # Episodes alone make a non-empty packet.
    assert prompt.startswith("<haki_memory>")
