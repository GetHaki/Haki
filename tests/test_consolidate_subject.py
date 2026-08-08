"""POST /v1/consolidate/subject: the console Playground's real "Write"
trigger — scoped to one project/subject, unlike the unscoped ops
POST /v1/consolidate. Verified by seeding two subjects' pending jobs and
checking only the requested one gets processed."""

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event

PROJECT = "prj_support"


async def test_consolidate_subject_only_processes_the_requested_scope(client):
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "fr"})], subject_id="usr_a")])
    await capture(client, [make_memory_event([mock_fact("language", {"lang": "en"})], subject_id="usr_b")])

    response = await client.post(
        "/v1/consolidate/subject",
        params={"project_id": PROJECT, "subject_id": "usr_a"},
    )
    assert response.status_code == 200
    assert response.json()["processed"] == 1

    facts_a = await client.get(
        "/v1/facts", params={"project_id": PROJECT, "subject_id": "usr_a"}
    )
    assert len(facts_a.json()["facts"]) == 1

    # usr_b's job is untouched — still pending, no fact yet.
    facts_b = await client.get(
        "/v1/facts", params={"project_id": PROJECT, "subject_id": "usr_b"}
    )
    assert facts_b.json()["facts"] == []

    # Running it again for usr_a processes nothing more (job already done).
    again = await client.post(
        "/v1/consolidate/subject",
        params={"project_id": PROJECT, "subject_id": "usr_a"},
    )
    assert again.json()["processed"] == 0


async def test_consolidate_subject_requires_scope_params(client):
    response = await client.post("/v1/consolidate/subject", params={"project_id": PROJECT})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "missing_scope"


async def test_consolidate_subject_with_no_pending_jobs_is_a_noop(client):
    response = await client.post(
        "/v1/consolidate/subject",
        params={"project_id": PROJECT, "subject_id": "usr_nobody"},
    )
    assert response.status_code == 200
    assert response.json()["processed"] == 0
