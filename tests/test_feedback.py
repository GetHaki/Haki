"""POST /v1/feedback (sprint 6): recording, validation, and the
`incorrect` -> disputed transition that removes the fact from context.
"""

from sqlalchemy import select

from app.db import async_session
from app.models import Fact, FactStatus, Feedback
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker


async def _active_fact_id(client) -> str:
    await capture(
        client, [make_memory_event([mock_fact("invoice_language", {"language": "fr"})])]
    )
    await run_worker()
    async with async_session() as session:
        fact = (
            await session.execute(
                select(Fact).where(Fact.predicate == "invoice_language")
            )
        ).scalars().one()
    assert fact.status is FactStatus.active
    return str(fact.id)


async def test_feedback_incorrect_disputes_fact_and_context_stops_serving_it(client):
    fact_id = await _active_fact_id(client)

    # Before feedback: the fact is served.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "invoice_language"},
    )
    assert [f["id"] for f in response.json()["packet"]["facts"]] == [fact_id]

    response = await client.post(
        "/v1/feedback",
        json={
            "project_id": "prj_support",
            "fact_id": fact_id,
            "rating": "incorrect",
            "comment": "la langue réelle est l'anglais",
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "recorded"
    assert response.json()["fact_status"] == "disputed"

    async with async_session() as session:
        stored = (await session.execute(select(Feedback))).scalars().all()
    assert len(stored) == 1
    assert stored[0].rating == "incorrect"

    # A disputed fact is never served as active again.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "invoice_language"},
    )
    assert response.json()["packet"]["facts"] == []


async def test_feedback_useful_keeps_fact_active(client):
    fact_id = await _active_fact_id(client)
    response = await client.post(
        "/v1/feedback",
        json={"project_id": "prj_support", "fact_id": fact_id, "rating": "useful"},
    )
    assert response.status_code == 201
    assert response.json()["fact_status"] == "active"


async def test_feedback_requires_exactly_one_target(client):
    both = await client.post(
        "/v1/feedback",
        json={
            "project_id": "prj_support",
            "trace_id": "00000000-0000-0000-0000-000000000001",
            "fact_id": "00000000-0000-0000-0000-000000000002",
            "rating": "useful",
        },
    )
    assert both.status_code == 422
    assert both.json()["error"]["type"] == "invalid_payload"

    none = await client.post(
        "/v1/feedback", json={"project_id": "prj_support", "rating": "useful"}
    )
    assert none.status_code == 422
    assert none.json()["error"]["type"] == "invalid_payload"


async def test_feedback_on_fact_of_another_project_is_a_plain_404(client):
    fact_id = await _active_fact_id(client)
    response = await client.post(
        "/v1/feedback",
        json={"project_id": "prj_other", "fact_id": fact_id, "rating": "incorrect"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "fact_not_found"

    # And the fact is untouched.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "invoice_language"},
    )
    assert [f["id"] for f in response.json()["packet"]["facts"]] == [fact_id]
