"""Inspection listings for the console (sprint 9): /v1/facts and /v1/traces.

Both are read-only, scope-bound listings. Behaviors verified: scope
isolation, content and ordering, mandatory scope params, API key auth.
"""

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_support"
SUBJECT = "usr_42"


async def _seed_facts(client) -> None:
    """One active fact for usr_42, one for another subject, same project."""
    await capture(
        client,
        [
            make_memory_event([mock_fact("invoice_language", {"language": "fr"})]),
            make_memory_event(
                [mock_fact("invoice_language", {"language": "en"}, subject_id="usr_99")],
                subject_id="usr_99",
            ),
        ],
    )
    assert await run_worker() == 1


async def test_facts_list_returns_facts_with_provenance(client):
    await _seed_facts(client)

    response = await client.get(
        "/v1/facts", params={"project_id": PROJECT, "subject_id": SUBJECT}
    )
    assert response.status_code == 200
    facts = response.json()["facts"]
    assert len(facts) == 1
    fact = facts[0]
    assert fact["predicate"] == "invoice_language"
    assert fact["value"] == {"language": "fr"}
    assert fact["status"] == "active"
    assert fact["version"] >= 1
    assert fact["subject_id"] == SUBJECT
    assert len(fact["source_event_ids"]) == 1
    assert fact["recorded_from"] is not None


async def test_facts_list_is_scoped_and_filters_by_status(client):
    await _seed_facts(client)

    # Another subject of the same project sees only its own facts.
    other = await client.get(
        "/v1/facts", params={"project_id": PROJECT, "subject_id": "usr_99"}
    )
    assert {f["value"]["language"] for f in other.json()["facts"]} == {"en"}

    # The status filter narrows the list.
    active = await client.get(
        "/v1/facts",
        params={"project_id": PROJECT, "subject_id": SUBJECT, "status": "active"},
    )
    assert len(active.json()["facts"]) == 1
    superseded = await client.get(
        "/v1/facts",
        params={
            "project_id": PROJECT,
            "subject_id": SUBJECT,
            "status": "superseded",
        },
    )
    assert superseded.json()["facts"] == []

    # Scope params are mandatory.
    missing = await client.get("/v1/facts", params={"project_id": PROJECT})
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "missing_scope"


async def test_facts_requires_a_valid_api_key(client, auth_required, make_api_key):
    auth_required.auth_required = False  # seed without auth, then enforce it
    await _seed_facts(client)
    auth_required.auth_required = True
    key = await make_api_key(project_id=PROJECT, org_id="org_acme")
    params = {"project_id": PROJECT, "subject_id": SUBJECT}

    anonymous = await client.get("/v1/facts", params=params)
    assert anonymous.status_code == 401

    authed = await client.get(
        "/v1/facts", params=params, headers={"Authorization": f"Bearer {key}"}
    )
    assert authed.status_code == 200
    assert len(authed.json()["facts"]) == 1

    # A key bound to another project cannot read this scope (403, no leak).
    foreign = await make_api_key(project_id="prj_other", org_id="org_other")
    denied = await client.get(
        "/v1/facts", params=params, headers={"Authorization": f"Bearer {foreign}"}
    )
    assert denied.status_code == 403


async def _make_trace(client, subject_id: str = SUBJECT, query: str = "langue ?"):
    response = await client.post(
        "/v1/context",
        json={
            "project_id": PROJECT,
            "subject_id": subject_id,
            "query": query,
            "purpose": "support",
        },
    )
    assert response.status_code == 200
    return response.json()["trace_id"]


async def test_traces_list_returns_recent_first(client):
    first_id = await _make_trace(client, query="première question")
    second_id = await _make_trace(client, query="seconde question")
    await _make_trace(client, subject_id="usr_99", query="autre sujet")

    response = await client.get("/v1/traces", params={"project_id": PROJECT})
    assert response.status_code == 200
    traces = response.json()["traces"]
    assert len(traces) == 3
    # Newest first: the second trace is ahead of the first.
    ids = [t["id"] for t in traces]
    assert ids.index(second_id) < ids.index(first_id)
    assert {t["query"] for t in traces} == {
        "première question",
        "seconde question",
        "autre sujet",
    }
    assert all(t["project_id"] == PROJECT for t in traces)

    # The subject filter narrows the list.
    scoped = await client.get(
        "/v1/traces", params={"project_id": PROJECT, "subject_id": SUBJECT}
    )
    assert len(scoped.json()["traces"]) == 2
    assert all(t["subject_id"] == SUBJECT for t in scoped.json()["traces"])

    # Scope params are mandatory.
    missing = await client.get("/v1/traces")
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "missing_scope"


async def test_traces_requires_a_valid_api_key(client, auth_required, make_api_key):
    auth_required.auth_required = False  # seed without auth, then enforce it
    await _make_trace(client)
    auth_required.auth_required = True
    params = {"project_id": PROJECT}

    anonymous = await client.get("/v1/traces", params=params)
    assert anonymous.status_code == 401

    key = await make_api_key(project_id=PROJECT, org_id="org_acme")
    authed = await client.get(
        "/v1/traces", params=params, headers={"Authorization": f"Bearer {key}"}
    )
    assert authed.status_code == 200
    assert len(authed.json()["traces"]) == 1
