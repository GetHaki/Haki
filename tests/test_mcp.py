"""MCP server end-to-end (sprint 4): real uvicorn subprocess on a free
port, official MCP client (mcp.client.streamable_http).

Scenario: a convention is captured -> haki_context recalls it ->
haki_inspect shows the decision trace -> haki_forget erases it -> the next
haki_context serves nothing. Plus the dev bearer auth on /mcp.

Note on seeding: the test providers are fake (hermetic, no network), and
the fake extractor reads `payload["mock_facts"]` — which the haki_capture
tool payload does not carry (its contract is content/subject_id/kind only,
per the PRD). The recallable convention is therefore seeded through the
HTTP API (POST /v1/capture + /v1/consolidate) against the SAME database
and project the MCP server is configured with; haki_capture itself is
verified through the timeline (event stored, consolidation attempted).
The live extraction path (real LLM through haki_capture) is covered by the
sprint-4 live demo.
"""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

ROOT = Path(__file__).resolve().parent.parent
MCP_PROJECT = "prj_mcp_test"
SUBJECT = "usr_mcp"
CONVENTION = "convention.tests"
CONVENTION_VALUE = {"rule": "tests avant toute modification"}
ADMIN_KEY = "admin-secret-for-mcp-tests"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server_env(api_key: str | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "HAKI_DATABASE_URL": "postgresql+asyncpg://haki:haki@localhost:5433/haki_test",
        "HAKI_LLM_PROVIDER": "fake",
        "HAKI_EMBED_PROVIDER": "fake",
        "HAKI_MCP_PROJECT_ID": MCP_PROJECT,
        "HAKI_MCP_ORG_ID": "org_mcp_test",
        "HAKI_MCP_SUBJECT_ID": SUBJECT,
        "HAKI_MCP_AUTOCONSOLIDATE": "true",
        "HAKI_ADMIN_KEY": ADMIN_KEY,
    }
    if api_key:
        env["HAKI_API_KEY"] = api_key
    else:
        env.pop("HAKI_API_KEY", None)
    return env


def _start_server(port: int, api_key: str | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=_server_env(api_key),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"MCP server exited early (code {proc.returncode})")
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    proc.kill()
    raise RuntimeError("MCP server did not become healthy in 60 s")


@pytest.fixture(scope="module")
def mcp_server(migrated_database):
    """uvicorn subprocess serving the Haki app (MCP on /mcp), open mode."""
    port = _free_port()
    proc = _start_server(port)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def mcp_server_with_key(migrated_database):
    port = _free_port()
    proc = _start_server(port, api_key="test-secret-key")
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


def _tool_data(result) -> dict:
    """Structured content of a tool result (fall back to JSON text)."""
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def _call(base_url: str, tool: str, arguments: dict) -> dict:
    async with streamable_http_client(f"{base_url}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return _tool_data(result)


async def _seed_convention(base_url: str) -> None:
    """Capture a convention via the HTTP API (mock_facts) + consolidate."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as http:
        response = await http.post(
            "/v1/capture",
            json={
                "idempotency_key": f"mcp-test-{uuid.uuid4()}",
                "events": [
                    {
                        "org_id": "org_mcp_test",
                        "project_id": MCP_PROJECT,
                        "subject_type": "user",
                        "subject_id": SUBJECT,
                        "kind": "conversation.message",
                        "occurred_at": "2026-08-01T10:00:00Z",
                        "payload": {
                            "role": "user",
                            "content": "Convention du projet.",
                            "mock_facts": [
                                {
                                    "subject_id": SUBJECT,
                                    "predicate": CONVENTION,
                                    "value": CONVENTION_VALUE,
                                    "action": "create",
                                    "confidence": 0.95,
                                }
                            ],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 202
        response = await http.post("/v1/consolidate")
        assert response.status_code == 200
        assert response.json()["processed"] == 1


async def test_mcp_full_scenario(mcp_server):
    await _seed_convention(mcp_server)

    async with streamable_http_client(f"{mcp_server}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {
                "haki_context",
                "haki_capture",
                "haki_inspect",
                "haki_forget",
                "haki_correct",
            }

            # 1. haki_capture: the event is stored (consolidation attempted).
            # No subject_id argument: the tool is locked to HAKI_MCP_SUBJECT_ID
            # (security invariant, the model never chooses scopes).
            capture_result = await session.call_tool(
                "haki_capture",
                {"content": "Decision: Postgres 16 + pgvector."},
            )
            capture_data = _tool_data(capture_result)
            assert uuid.UUID(capture_data["event_id"])
            assert capture_data["project_id"] == MCP_PROJECT

            # M8: content captured via MCP is written by the MODEL, not the
            # human — the event is stamped semi_trusted, hardcoded server-
            # side (no MCP tool ever takes a trust parameter).
            async with httpx.AsyncClient(base_url=mcp_server, timeout=10) as http:
                timeline_response = await http.get(
                    "/v1/timeline",
                    params={"project_id": MCP_PROJECT, "subject_id": SUBJECT},
                )
            captured_event = next(
                e
                for e in timeline_response.json()["events"]
                if e["id"] == capture_data["event_id"]
            )
            assert captured_event["origin_trust"] == "semi_trusted"

            # 2. haki_context recalls the seeded convention, formatted.
            context_result = await session.call_tool(
                "haki_context",
                {"query": "quelles conventions avant de modifier du code ?"},
            )
            context_data = _tool_data(context_result)
            assert CONVENTION in context_data["context"]
            assert "tests avant toute modification" in context_data["context"]
            assert "source:" in context_data["context"]
            trace_id = context_data["trace_id"]
            predicates = [fact["predicate"] for fact in context_data["facts"]]
            assert CONVENTION in predicates

            # 3. haki_inspect: the trace explains the inclusion.
            inspect_result = await session.call_tool(
                "haki_inspect", {"trace_id": trace_id}
            )
            inspect_data = _tool_data(inspect_result)
            assert inspect_data["query"] == (
                "quelles conventions avant de modifier du code ?"
            )
            included = [
                d for d in inspect_data["decisions"] if d["action"] == "included"
            ]
            assert included, inspect_data["decisions"]

            # 4. haki_forget (delete): the convention disappears.
            forget_result = await session.call_tool(
                "haki_forget", {"mode": "delete"}
            )
            forget_data = _tool_data(forget_result)
            assert forget_data["status"] == "ok"
            assert forget_data["facts_deleted"] >= 1
            assert uuid.UUID(forget_data["forget_id"])

            after_result = await session.call_tool(
                "haki_context",
                {"query": "quelles conventions avant de modifier du code ?"},
            )
            after_data = _tool_data(after_result)
            assert after_data["facts"] == []
            assert CONVENTION not in after_data["context"]

    # haki_capture really stored the event (visible on the HTTP timeline).
    async with httpx.AsyncClient(base_url=mcp_server, timeout=10) as http:
        response = await http.get(
            "/v1/timeline",
            params={"project_id": MCP_PROJECT, "subject_id": SUBJECT},
        )
    # mode=delete erased the subject's events too: the timeline is empty,
    # which also proves the capture had landed (forget deleted >= 1 event).
    assert response.status_code == 200
    assert response.json()["events"] == []


async def test_mcp_capture_is_idempotent(mcp_server):
    args = {"content": "Convention: commits en francais."}
    first = await _call(mcp_server, "haki_capture", args)
    assert first["deduplicated"] is False
    second = await _call(mcp_server, "haki_capture", args)
    assert second["deduplicated"] is True
    assert second["event_id"] == first["event_id"]


async def test_mcp_ignores_a_client_supplied_subject_id(mcp_server):
    """Security invariant (README): the model never chooses scopes. Even if
    a client passes a subject_id argument (a malicious/confused agent, or a
    stale client built against the pre-fix tool signature), the tool must
    keep serving the server-configured subject (HAKI_MCP_SUBJECT_ID) and
    never the client-supplied one."""
    result = await _call(
        mcp_server, "haki_context", {"query": "x", "subject_id": "attacker_chosen"}
    )
    assert f"sujet {SUBJECT}" in result["context"]
    assert "attacker_chosen" not in result["context"]


async def test_mcp_correct_matches_direct_feedback_call(mcp_server):
    """haki_correct (M10) must have the exact same effect as a direct
    POST /v1/feedback call: rating=incorrect on a fact_id disputes it and
    the Context Assembler stops serving it — proven here by disputing two
    equivalent facts, one through each path, and observing identical
    outcomes."""
    async with httpx.AsyncClient(base_url=mcp_server, timeout=30) as http:
        response = await http.post(
            "/v1/capture",
            json={
                "idempotency_key": f"mcp-correct-{uuid.uuid4()}",
                "events": [
                    {
                        "org_id": "org_mcp_test",
                        "project_id": MCP_PROJECT,
                        "subject_type": "user",
                        "subject_id": SUBJECT,
                        "kind": "conversation.message",
                        "occurred_at": "2026-08-01T10:00:00Z",
                        "payload": {
                            "role": "user",
                            "content": "faits a corriger",
                            "mock_facts": [
                                {
                                    "subject_id": SUBJECT,
                                    "predicate": "invoice_language_via_mcp",
                                    "value": {"language": "es"},
                                    "action": "create",
                                    "confidence": 0.9,
                                },
                                {
                                    "subject_id": SUBJECT,
                                    "predicate": "invoice_language_via_http",
                                    "value": {"language": "es"},
                                    "action": "create",
                                    "confidence": 0.9,
                                },
                            ],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 202
        response = await http.post("/v1/consolidate")
        assert response.status_code == 200
        assert response.json()["processed"] == 1

        context_mcp = await http.post(
            "/v1/context",
            json={
                "project_id": MCP_PROJECT,
                "subject_id": SUBJECT,
                "query": "invoice_language_via_mcp",
            },
        )
        mcp_fact_id = context_mcp.json()["packet"]["facts"][0]["id"]

        context_http = await http.post(
            "/v1/context",
            json={
                "project_id": MCP_PROJECT,
                "subject_id": SUBJECT,
                "query": "invoice_language_via_http",
            },
        )
        http_fact_id = context_http.json()["packet"]["facts"][0]["id"]

        # Baseline: a direct HTTP call to /v1/feedback.
        direct = await http.post(
            "/v1/feedback",
            json={
                "project_id": MCP_PROJECT,
                "fact_id": http_fact_id,
                "rating": "incorrect",
                "comment": "langue reelle: francais",
            },
        )
        assert direct.status_code == 201
        assert direct.json()["status"] == "recorded"
        assert direct.json()["fact_status"] == "disputed"

    # Same correction, this time through the haki_correct MCP tool.
    mcp_result = await _call(
        mcp_server,
        "haki_correct",
        {
            "fact_id": mcp_fact_id,
            "rating": "incorrect",
            "comment": "langue reelle: francais",
        },
    )
    assert mcp_result["status"] == "recorded"
    assert mcp_result["fact_status"] == "disputed"
    assert uuid.UUID(mcp_result["feedback_id"])

    # Identical effect: both facts are equally never served again.
    async with httpx.AsyncClient(base_url=mcp_server, timeout=30) as http:
        after_mcp = await http.post(
            "/v1/context",
            json={
                "project_id": MCP_PROJECT,
                "subject_id": SUBJECT,
                "query": "invoice_language_via_mcp",
            },
        )
        after_http = await http.post(
            "/v1/context",
            json={
                "project_id": MCP_PROJECT,
                "subject_id": SUBJECT,
                "query": "invoice_language_via_http",
            },
        )
    assert after_mcp.json()["packet"]["facts"] == []
    assert after_http.json()["packet"]["facts"] == []


async def test_mcp_correct_requires_exactly_one_target(mcp_server):
    """Same validation as POST /v1/feedback (app.ledger.submit_feedback):
    neither fact_id nor trace_id is refused loudly — a typed MCP tool
    error (is_error=True), never a silent no-op or a server crash."""
    async with streamable_http_client(f"{mcp_server}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("haki_correct", {"rating": "useful"})
    assert result.is_error
    assert "trace_id or fact_id" in result.content[0].text


async def _create_key(base_url: str, *, org_id: str, project_id: str) -> str:
    """Mints a key for an arbitrary org/project via the admin key (the test
    servers all set HAKI_ADMIN_KEY — see _server_env) rather than relying
    on the anonymous "first key is free" bootstrap, which only ever
    applies once per server and would fail on the second call in a test
    file that mints several distinct tenants."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as http:
        response = await http.post(
            "/v1/keys",
            json={"org_id": org_id, "project_id": project_id},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        assert response.status_code == 201, response.text
        return response.json()["key"]


async def test_mcp_scopes_by_real_customer_api_key(mcp_server):
    """The production-readiness gap found live: the shared Cloud /mcp
    endpoint used to serve EVERY caller from the single HAKI_MCP_* server
    config, so two different signed-up customers on the same running
    server could never be told apart. A real hk_ key (Authorization
    header) + a client-set X-Haki-Subject-Id header must now resolve to
    THAT key's own org/project — different from, and isolated from, the
    server's legacy default scope (MCP_PROJECT/SUBJECT used by every other
    test in this file, which exercises the no-Authorization-header
    self-hosted fallback path instead)."""
    tenant_project = "prj_mcp_tenant_a"
    tenant_subject = "usr_tenant_a"
    key = await _create_key(mcp_server, org_id="org_mcp_tenant_a", project_id=tenant_project)

    async with create_mcp_http_client(
        headers={"Authorization": f"Bearer {key}", "X-Haki-Subject-Id": tenant_subject}
    ) as http_client:
        async with streamable_http_client(f"{mcp_server}/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                capture = await session.call_tool(
                    "haki_capture", {"content": "Decision propre a ce tenant."}
                )
    capture_data = _tool_data(capture)
    assert capture_data["project_id"] == tenant_project
    assert capture_data["project_id"] != MCP_PROJECT  # never the legacy default

    # Visible on the tenant's own project/subject via the HTTP API — proof
    # the write actually landed in the resolved scope, not the default one.
    async with httpx.AsyncClient(base_url=mcp_server, timeout=10) as http:
        response = await http.get(
            "/v1/timeline",
            params={"project_id": tenant_project, "subject_id": tenant_subject},
        )
    assert response.status_code == 200
    assert len(response.json()["events"]) == 1


async def test_mcp_two_tenants_on_the_same_server_never_cross(mcp_server):
    """The core multi-tenancy claim: two different customers' keys, same
    running /mcp process, never see each other's memory."""
    project_a = await _create_key(mcp_server, org_id="org_mcp_x", project_id="prj_mcp_x")
    project_b = await _create_key(mcp_server, org_id="org_mcp_y", project_id="prj_mcp_y")

    async def _capture_as(key: str) -> dict:
        async with create_mcp_http_client(
            headers={"Authorization": f"Bearer {key}", "X-Haki-Subject-Id": "usr_shared"}
        ) as http_client:
            async with streamable_http_client(f"{mcp_server}/mcp", http_client=http_client) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "haki_context", {"query": "quoi que ce soit"}
                    )
        return _tool_data(result)

    result_a = await _capture_as(project_a)
    result_b = await _capture_as(project_b)
    assert result_a["project_id"] == "prj_mcp_x"
    assert result_b["project_id"] == "prj_mcp_y"


async def test_mcp_rejects_invalid_api_key(mcp_server):
    async with create_mcp_http_client(
        headers={"Authorization": "Bearer hk_not_a_real_key", "X-Haki-Subject-Id": "usr_x"}
    ) as http_client:
        async with streamable_http_client(f"{mcp_server}/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("haki_context", {"query": "x"})
    assert result.is_error
    assert "invalid or revoked" in result.content[0].text


async def test_mcp_requires_subject_header_with_a_real_key(mcp_server):
    key = await _create_key(mcp_server, org_id="org_mcp_noheader", project_id="prj_mcp_noheader")
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {key}"}) as http_client:
        async with streamable_http_client(f"{mcp_server}/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("haki_context", {"query": "x"})
    assert result.is_error
    assert "X-Haki-Subject-Id" in result.content[0].text


async def test_mcp_real_key_bypasses_the_legacy_shared_secret(mcp_server_with_key):
    """Found live: when HAKI_API_KEY (the legacy single shared secret) is
    configured — exactly the production setup — a real customer's hk_ key
    used to be rejected outright by the middleware before even reaching
    the tool layer, because it never equals that one shared value. A real
    hk_ key must work on its own, without also needing the shared secret."""
    key = await _create_key(
        mcp_server_with_key, org_id="org_mcp_withkey", project_id="prj_mcp_withkey"
    )
    async with create_mcp_http_client(
        headers={"Authorization": f"Bearer {key}", "X-Haki-Subject-Id": "usr_withkey"}
    ) as http_client:
        async with streamable_http_client(
            f"{mcp_server_with_key}/mcp", http_client=http_client
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("haki_context", {"query": "x"})
    assert not result.is_error, result.content


async def test_mcp_dev_auth(mcp_server_with_key):
    # No header -> 401 before any MCP processing.
    response = httpx.post(
        f"{mcp_server_with_key}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"

    # Bearer header -> full MCP handshake works.
    async with create_mcp_http_client(
        headers={"Authorization": "Bearer test-secret-key"}
    ) as http_client:
        async with streamable_http_client(
            f"{mcp_server_with_key}/mcp", http_client=http_client
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
    assert {tool.name for tool in tools.tools} == {
        "haki_context",
        "haki_capture",
        "haki_inspect",
        "haki_forget",
        "haki_correct",
    }
