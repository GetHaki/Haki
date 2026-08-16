"""Basic rate limiting: a caller that exceeds a route's limit gets a real
429, served by slowapi's real in-memory backend (never mocked -- same
convention as the rest of this suite, see tests/conftest.py's
reset_rate_limiter).
"""


async def test_context_over_limit_returns_429(client):
    """POST /v1/context had no rate limit at all before this (found by
    security review, 16 aout) -- build_context makes no LLM call so a
    leaked key couldn't run up a bill here, but the query itself (hybrid
    scoring, multi-hop expansion) is DB-expensive and was completely
    unthrottled. 120/minute, see app/api/routes/context.py."""
    body = {"project_id": "prj_rate_limit_ctx", "subject_id": "usr_1", "query": "topic"}

    responses = [await client.post("/v1/context", json=body) for _ in range(120)]
    assert all(r.status_code == 200 for r in responses)

    blocked = await client.post("/v1/context", json=body)
    assert blocked.status_code == 429
    error = blocked.json()["error"]
    assert error["type"] == "rate_limited"
    assert "Retry-After" in blocked.headers
