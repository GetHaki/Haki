"""POST /v1/conflicts/{id}/resolve (sprint 6): keep one fact, supersede
the others, close the set — then context serves the kept fact again.
"""

import uuid

from sqlalchemy import select

from app.db import async_session
from app.models import ConflictSet, Fact, FactStatus
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


async def _open_conflict(client) -> tuple[ConflictSet, list[Fact]]:
    """Two contradictory facts (language fr vs en) in one open conflict set."""
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})])])
    await run_worker()
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "en"})])])
    await run_worker()
    async with async_session() as session:
        conflict = (await session.execute(select(ConflictSet))).scalars().one()
        conflict_id = conflict.id
    facts = await facts_for("usr_42", "language")
    async with async_session() as session:
        conflict = await session.get(ConflictSet, conflict_id)
        # Detach a plain snapshot for assertions.
        snapshot = ConflictSet(
            id=conflict.id,
            project_id=conflict.project_id,
            subject_id=conflict.subject_id,
            fact_ids=list(conflict.fact_ids),
            status=conflict.status,
        )
    return snapshot, facts


async def test_resolve_keeps_one_fact_supersedes_the_other_and_closes_set(client):
    conflict, facts = await _open_conflict(client)
    kept = next(f for f in facts if f.value == {"lang": "en"})
    loser = next(f for f in facts if f.value == {"lang": "fr"})

    # While open, context serves nothing (conflict_open).
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "language"},
    )
    assert response.json()["packet"]["facts"] == []

    response = await client.post(
        f"/v1/conflicts/{conflict.id}/resolve",
        json={"project_id": "prj_support", "keep_fact_id": str(kept.id)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resolved"
    assert body["kept_fact_id"] == str(kept.id)
    assert body["superseded_fact_ids"] == [str(loser.id)]
    assert body["resolved_at"]

    async with async_session() as session:
        kept_row = await session.get(Fact, kept.id)
        loser_row = await session.get(Fact, loser.id)
        conflict_row = await session.get(ConflictSet, conflict.id)
    assert kept_row.status is FactStatus.active
    assert loser_row.status is FactStatus.superseded
    assert loser_row.supersedes_id == kept.id
    assert conflict_row.status == "resolved"
    assert conflict_row.resolved_at is not None

    # After resolution, context serves the kept fact — and only it.
    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "language"},
    )
    served = response.json()["packet"]["facts"]
    assert [f["id"] for f in served] == [str(kept.id)]

    # The set no longer shows up as open.
    response = await client.get("/v1/conflicts", params={"project_id": "prj_support"})
    body = response.json()
    assert body["conflicts"] == []
    assert body["open_count"] == 0
    assert body["oldest_open_seconds"] is None


async def test_open_conflicts_summary_reports_count_and_oldest_age(client):
    """Observability (sprint 10): open conflicts are never auto-resolved
    (hide-both is deliberate), so the list endpoint carries a summary a
    monitoring job can alert on without parsing the full list."""
    await _open_conflict(client)

    response = await client.get("/v1/conflicts", params={"project_id": "prj_support"})
    body = response.json()
    assert body["open_count"] == 1
    assert body["oldest_open_seconds"] is not None
    assert body["oldest_open_seconds"] >= 0


async def test_resolve_with_fact_outside_the_set_is_a_typed_error(client):
    conflict, _ = await _open_conflict(client)
    response = await client.post(
        f"/v1/conflicts/{conflict.id}/resolve",
        json={"project_id": "prj_support", "keep_fact_id": str(uuid.uuid4())},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "fact_not_in_conflict"


async def test_resolve_twice_is_a_typed_error(client):
    conflict, facts = await _open_conflict(client)
    kept = next(f for f in facts if f.value == {"lang": "en"})
    payload = {"project_id": "prj_support", "keep_fact_id": str(kept.id)}
    first = await client.post(f"/v1/conflicts/{conflict.id}/resolve", json=payload)
    assert first.status_code == 200
    second = await client.post(f"/v1/conflicts/{conflict.id}/resolve", json=payload)
    assert second.status_code == 409
    assert second.json()["error"]["type"] == "conflict_already_resolved"


async def test_resolve_with_wrong_project_is_a_plain_404(client):
    conflict, facts = await _open_conflict(client)
    kept = next(f for f in facts if f.value == {"lang": "en"})
    response = await client.post(
        f"/v1/conflicts/{conflict.id}/resolve",
        json={"project_id": "prj_other", "keep_fact_id": str(kept.id)},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "conflict_not_found"
