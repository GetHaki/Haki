"""GET /v1/projects and GET /v1/subjects (console "Projects & subjects").

/v1/projects deliberately spans every project of the caller's own org (an
org can have several projects — self-hosted/curl callers already do via
free-string project_ids on POST /v1/keys) while never leaking another
org's projects. /v1/subjects stays single-project like /v1/facts.
"""

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

ORG = "org_acme"


async def _seed(client, project_id: str, subject_id: str) -> None:
    event = make_memory_event(
        [mock_fact("invoice_language", {"language": "fr"}, subject_id=subject_id)],
        subject_id=subject_id,
    )
    event["project_id"] = project_id
    await capture(client, [event])
    assert await run_worker() == 1


async def test_projects_lists_only_the_callers_org_with_counts(
    client, auth_required, make_api_key
):
    auth_required.auth_required = False
    await _seed(client, "prj_support", "usr_1")
    await _seed(client, "prj_ops", "usr_2")
    await _seed(client, "prj_other_org", "usr_9")  # different org below
    auth_required.auth_required = True

    await make_api_key(project_id="prj_support", org_id=ORG)
    await make_api_key(project_id="prj_ops", org_id=ORG)
    key = await make_api_key(project_id="prj_support", org_id=ORG, label="second")
    await make_api_key(project_id="prj_other_org", org_id="org_other")

    response = await client.get(
        "/v1/projects", headers={"Authorization": f"Bearer {key}"}
    )
    assert response.status_code == 200
    projects = {p["project_id"]: p for p in response.json()["projects"]}

    assert set(projects) == {"prj_support", "prj_ops"}
    assert projects["prj_support"]["active_facts"] == 1
    assert projects["prj_support"]["subjects"] == 1


async def test_projects_requires_a_valid_key(client, auth_required):
    response = await client.get("/v1/projects")
    assert response.status_code == 401


async def test_subjects_lists_facts_and_recalls_per_subject(
    client, auth_required, make_api_key
):
    auth_required.auth_required = False
    await _seed(client, "prj_support", "usr_1")
    await _seed(client, "prj_support", "usr_2")
    await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_1", "query": "langue ?"},
    )
    auth_required.auth_required = True

    key = await make_api_key(project_id="prj_support", org_id=ORG)
    response = await client.get(
        "/v1/subjects",
        params={"project_id": "prj_support"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200
    subjects = {s["subject_id"]: s for s in response.json()["subjects"]}
    assert set(subjects) == {"usr_1", "usr_2"}
    assert subjects["usr_1"]["facts"] == 1
    assert subjects["usr_1"]["recalls"] == 1
    assert subjects["usr_2"]["recalls"] == 0


async def test_subjects_requires_project_id(client):
    response = await client.get("/v1/subjects")
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "missing_scope"
