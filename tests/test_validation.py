"""Validation: typed errors for missing scope and malformed payloads."""

import uuid

from tests.test_capture import make_batch


async def test_capture_without_subject_id_returns_missing_scope(client):
    batch = make_batch(f"batch-{uuid.uuid4()}")
    del batch["events"][0]["subject_id"]

    response = await client.post("/v1/capture", json=batch)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "missing_scope"
    assert error["field"] == "events.0.subject_id"


async def test_capture_malformed_payload_returns_invalid_payload(client):
    batch = make_batch(f"batch-{uuid.uuid4()}")
    batch["events"][0]["payload"] = "not-a-dict"

    response = await client.post("/v1/capture", json=batch)
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_payload"
