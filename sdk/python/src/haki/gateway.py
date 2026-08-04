"""Gateway helper: a preconfigured httpx client for the Haki Gateway.

The simplest real usage is the official OpenAI SDK — only `base_url`, the
Haki key and one header change:

    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8100/gateway/v1",
        api_key="hk_...",
        default_headers={"X-Haki-Subject-Id": "usr_42"},
    )
    client.chat.completions.create(model="...", messages=[...])

For plain httpx users, `gateway_client(...)` returns a client with the
Authorization and X-Haki-* headers already set:

    from haki.gateway import gateway_client

    client = gateway_client("http://localhost:8100/gateway/v1", "hk_...", "usr_42")
    response = client.post("/chat/completions", json={"model": "...", "messages": [...]})

Memory identity (subject, thread, run, purpose) travels in headers, never
in the request body: the model never chooses the scope it remembers.
"""

import httpx

__all__ = ["async_gateway_client", "gateway_client"]


def _headers(
    api_key: str,
    subject_id: str,
    *,
    thread_id: str | None,
    run_id: str | None,
    purpose: str | None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Haki-Subject-Id": subject_id,
    }
    if thread_id:
        headers["X-Haki-Thread-Id"] = thread_id
    if run_id:
        headers["X-Haki-Run-Id"] = run_id
    if purpose:
        headers["X-Haki-Purpose"] = purpose
    return headers


def gateway_client(
    base_url: str,
    api_key: str,
    subject_id: str,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    purpose: str | None = None,
    timeout: float = 120.0,
) -> httpx.Client:
    """Synchronous httpx client preconfigured for the Haki Gateway.

    `base_url` points at the gateway root (e.g.
    "http://localhost:8100/gateway/v1"); call
    `client.post("/chat/completions", json=...)` with a standard OpenAI
    body.
    """
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=_headers(
            api_key, subject_id, thread_id=thread_id, run_id=run_id, purpose=purpose
        ),
        timeout=timeout,
    )


def async_gateway_client(
    base_url: str,
    api_key: str,
    subject_id: str,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    purpose: str | None = None,
    timeout: float = 120.0,
) -> httpx.AsyncClient:
    """Asynchronous variant of `gateway_client`."""
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=_headers(
            api_key, subject_id, thread_id=thread_id, run_id=run_id, purpose=purpose
        ),
        timeout=timeout,
    )
