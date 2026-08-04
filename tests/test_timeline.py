"""Timeline: scope isolation and mandatory scope parameters."""

from tests.test_capture import make_batch


async def test_timeline_never_leaks_across_subjects(client):
    batch_a = make_batch("batch-scope-a")
    batch_b = make_batch("batch-scope-b")
    for event in batch_b["events"]:
        event["subject_id"] = "usr_99"

    await client.post("/v1/capture", json=batch_a)
    await client.post("/v1/capture", json=batch_b)

    timeline_a = await client.get(
        "/v1/timeline", params={"project_id": "prj_support", "subject_id": "usr_42"}
    )
    assert timeline_a.status_code == 200
    events_a = timeline_a.json()["events"]
    assert len(events_a) == 2
    assert all(e["subject_id"] == "usr_42" for e in events_a)

    timeline_b = await client.get(
        "/v1/timeline", params={"project_id": "prj_support", "subject_id": "usr_99"}
    )
    subjects_b = {e["subject_id"] for e in timeline_b.json()["events"]}
    assert subjects_b == {"usr_99"}


async def test_timeline_without_subject_id_is_rejected(client):
    response = await client.get("/v1/timeline", params={"project_id": "prj_support"})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "missing_scope"
    assert response.json()["error"]["field"] == "subject_id"
