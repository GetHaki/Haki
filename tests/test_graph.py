"""GET /v1/graph: nodes/edges derived from existing fact/event FKs, no
separate graph store — verified by seeding a real supersession chain."""

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_support"
SUBJECT = "usr_42"


async def test_graph_has_subject_and_fact_nodes_with_source_edge(client):
    await capture(
        client,
        [make_memory_event([mock_fact("invoice_language", {"language": "fr"})])],
    )
    assert await run_worker() == 1

    response = await client.get(
        "/v1/graph", params={"project_id": PROJECT, "subject_id": SUBJECT}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["subject_id"] == SUBJECT

    kinds = {n["kind"] for n in body["nodes"]}
    assert kinds == {"subject", "fact", "event"}

    subject_node = next(n for n in body["nodes"] if n["kind"] == "subject")
    assert subject_node["id"] == f"subject:{SUBJECT}"

    fact_node = next(n for n in body["nodes"] if n["kind"] == "fact")
    assert fact_node["status"] == "active"

    has_fact = [e for e in body["edges"] if e["kind"] == "has_fact"]
    assert len(has_fact) == 1
    assert has_fact[0] == {
        "source": subject_node["id"],
        "target": fact_node["id"],
        "kind": "has_fact",
    }

    derived_from = [e for e in body["edges"] if e["kind"] == "derived_from"]
    assert len(derived_from) == 1
    assert derived_from[0]["source"] == fact_node["id"]


async def test_graph_shows_supersession_edge(client):
    await capture(
        client,
        [make_memory_event([mock_fact("invoice_language", {"language": "fr"})])],
    )
    assert await run_worker() == 1
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("invoice_language", {"language": "en"}, action="supersede")]
            )
        ],
    )
    assert await run_worker() == 1

    response = await client.get(
        "/v1/graph", params={"project_id": PROJECT, "subject_id": SUBJECT}
    )
    body = response.json()
    fact_nodes = [n for n in body["nodes"] if n["kind"] == "fact"]
    assert len(fact_nodes) == 2

    supersedes = [e for e in body["edges"] if e["kind"] == "supersedes"]
    assert len(supersedes) == 1


async def test_graph_requires_scope_params(client):
    missing = await client.get("/v1/graph", params={"project_id": PROJECT})
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "missing_scope"
