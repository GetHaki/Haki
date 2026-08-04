"""Context Assembler behaviors (real database, FakeProvider):

token budget, scope isolation, trace inspection with reason codes.
"""

from tests.test_consolidator import capture, make_memory_event, run_worker
from app.providers.fake import mock_fact


async def test_budget_packs_best_scored_facts_and_traces_over_budget(client):
    # Three facts with large values (~50 estimated tokens each).
    big = {"detail": "x" * 200}
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact("topic_a", big),
                    mock_fact("topic_b", big),
                    mock_fact("topic_c", big),
                ]
            )
        ],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "topic",
            "budget_tokens": 60,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_count"] <= 60
    assert 1 <= len(body["packet"]["facts"]) < 3

    trace = await client.get(
        f"/v1/inspect/{body['trace_id']}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    decisions = trace.json()["decisions"]
    assert any(d["reason_code"] == "over_budget" for d in decisions)
    included = [d for d in decisions if d["action"] == "included"]
    assert len(included) == len(body["packet"]["facts"])


async def test_budget_zero_or_negative_is_a_typed_error(client):
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "anything",
            "budget_tokens": 0,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "budget_exceeded"


async def test_context_never_leaks_across_subjects(client):
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "pro"}, subject_id="usr_a")], subject_id="usr_a")],
    )
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "free"}, subject_id="usr_b")], subject_id="usr_b")],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_a", "query": "plan"},
    )
    served = response.json()["packet"]["facts"]
    assert len(served) == 1
    assert served[0]["value"] == {"tier": "pro"}


async def test_inspect_trace_never_leaks_across_scopes(client):
    await capture(
        client,
        [make_memory_event([mock_fact("plan", {"tier": "pro"})])],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={"project_id": "prj_support", "subject_id": "usr_42", "query": "plan"},
    )
    trace_id = response.json()["trace_id"]

    # Right scope: full trace with decisions.
    ok = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_support", "subject_id": "usr_42"},
    )
    assert ok.status_code == 200
    assert ok.json()["query"] == "plan"
    assert ok.json()["decisions"][0]["action"] == "included"

    # Other subject, same project: 404, never a leak of the trace.
    other_subject = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_support", "subject_id": "usr_99"},
    )
    assert other_subject.status_code == 404
    assert other_subject.json()["error"]["type"] == "trace_not_found"

    # Other project: 404 as well.
    other_project = await client.get(
        f"/v1/inspect/{trace_id}",
        params={"project_id": "prj_other", "subject_id": "usr_42"},
    )
    assert other_project.status_code == 404

    # Missing scope: typed error.
    missing = await client.get(f"/v1/inspect/{trace_id}")
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "missing_scope"
