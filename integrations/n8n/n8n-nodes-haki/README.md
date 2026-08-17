# n8n-nodes-haki

Community n8n nodes for [Haki](https://github.com/GetHaki/Haki) — long-term memory for AI agents.

- **Haki Context** — fetches the subject's ContextPacket (verified facts, warnings, `trace_id`). Always **before** the AI Agent.
- **Haki Capture** — records the user/assistant turn. Always **after** the AI Agent.

No Qdrant, Supabase, or Data Table to configure: both nodes talk directly to the Haki API (`POST /v1/context`, `POST /v1/capture`, `POST /v1/consolidate`).

## Installation (n8n self-hosted)

### Via the n8n UI

Settings → Community nodes → Install → `n8n-nodes-haki`.

### Via npm (self-hosted)

```bash
mkdir -p ~/.n8n/nodes
cd ~/.n8n/nodes
npm install n8n-nodes-haki
```

Then restart n8n. In Docker:

```bash
docker run -it --rm -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  --add-host host.docker.internal:host-gateway \
  n8nio/n8n
# then inside the container:
mkdir -p /home/node/.n8n/nodes && cd /home/node/.n8n/nodes && npm install n8n-nodes-haki
# restart the container to load the nodes
```

From this repository's source:

```bash
cd integrations/n8n/n8n-nodes-haki
npm install && npm run build && npm pack
cd ~/.n8n/nodes && npm install <path>/n8n-nodes-haki-0.2.0.tgz
```

> **n8n Cloud**: community nodes require a **verified** node there. Verification goes through the n8n Creator Portal (with GitHub Actions provenance) — submitted, not yet approved. Until then, on n8n Cloud: use the [native HTTP Request template](../haki-persistent-support-agent.json) instead.

## Configuration

One credential, **Haki API**:

| Field | Default | Note |
|---|---|---|
| Base URL | `http://localhost:8100` | n8n in Docker + API on the host: `http://host.docker.internal:8100` |
| API Key | *(empty)* | Optional in local dev; required once an admin key or API key is configured server-side |

## Haki Context

| Field | Required | Default | Note |
|---|---|---|---|
| Project ID | yes | — | Haki project (never chosen by the model) |
| Subject ID | yes | — | Stable identifier; **errors if empty or `default`** |
| Query | yes | — | The current message (used to rerank facts) |
| Budget Tokens | no | `2000` | ContextPacket token budget |
| Purpose | no | — | Task type, recorded in the trace |

Output: `context_text` (a `<haki_memory>` block ready to inject into the system prompt), `packet` (full JSON: facts + warnings), `trace_id`, `token_count`.

## Haki Capture

| Field | Required | Default | Note |
|---|---|---|---|
| Project ID | yes | — | Same project as Haki Context |
| Subject ID | yes | — | Reuse `{{ $('Haki Context').item.json.subject_id }}` |
| User Message | yes | — | The turn's user message |
| Assistant Message | yes | — | The agent's reply |
| Thread ID / Run ID | no | — | Feed the idempotency key |
| Wait Consolidation | no | `false` | When on: also calls `POST /v1/consolidate` (memory recallable immediately — dev/demo) |

**Idempotency**: the key is derived from `run_id` (or `thread_id`) plus a SHA-256 hash of the content (`n8n-turn-<run|thread>-<hash16>`). Re-running the same execution never duplicates the turn.

## Development

```bash
npm install
npm run build          # tsc -> dist/
npm test                # build + node:test harness against a real API (HAKI_BASE_URL, default http://localhost:8100)
```

The `test/run-tests.mjs` harness runs both compiled nodes' `execute()` with a minimal `IExecuteFunctions` mock, against a real Haki API: packet returned, `NodeOperationError` on an invalid subject, capture visible in `/v1/timeline`, idempotency on replay, synchronous consolidation.
