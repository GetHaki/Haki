# Haki × n8n

Two ways to give an n8n agent long-term memory, with no Qdrant/Supabase/Data Table to configure.

The required chain (PRD, Flow 2):

```text
Chat Trigger / Webhook → Haki Context → AI Agent → Haki Capture → Respond
```

**Haki Context always before the agent, Haki Capture always after.** `subject_id` is required and can never be empty or `default` — durable memory without a stable identity is dangerous memory.

## Option 1 — Native template (V1 beta, works everywhere)

[`haki-persistent-support-agent.json`](./haki-persistent-support-agent.json): an importable workflow using only the native **HTTP Request** node — no installation, works even on instances that can't install community nodes.

1. n8n → *Import from file* → the JSON.
2. Configure the **only 3 things** (detailed in the canvas sticky notes):
   - **Haki credential (header)** — Header Auth `Authorization: Bearer <key>` on both HTTP nodes (in local dev with no admin/API key configured, set authentication to *None*);
   - **LLM credential** — on *OpenAI Chat Model* (OpenRouter base URL pre-filled, editable);
   - **Subject mapping** — the template reads `{{ $json.body.subject_id }}`; adapt to your own source (sessionId, email, Telegram/WhatsApp ID...), never a shared constant.
3. Activate the workflow, then:

```bash
curl -X POST http://localhost:5678/webhook/haki-support-agent \
  -H 'content-type: application/json' \
  -d '{"subject_id": "usr_123", "message": "I prefer my replies in French"}'
```

The **IF "Valid subject?"** node rejects (HTTP 400) any call without a stable `subject_id`.

## Option 2 — Community node package (V1.1)

**[`n8n-nodes-haki`](https://github.com/GetHaki/n8n-nodes-haki)** (own repository, not part of this monorepo — n8n's Creator Portal verification requires a single-purpose repo structure): visual **Haki Context** / **Haki Capture** nodes + a **Haki API** credential (base URL + optional key). Built-in subject validation (a readable execution error if empty/`default`), `context_text` output ready to inject, idempotency derived from the run/thread, `wait_consolidation` for memory that's recallable immediately.

Installation and details: see the [package README](https://github.com/GetHaki/n8n-nodes-haki#readme). Published on npm; n8n Cloud additionally requires a **verified** node — automated review passed, pending manual review (demo video) on the Creator Portal.

## End-to-end verification

- Node harness outside n8n ([`n8n-nodes-haki/test/`](https://github.com/GetHaki/n8n-nodes-haki/tree/main/test)): 7/7 against a real API.
- Real execution in n8n Docker (`n8nio/n8n` 2.32.7, package mounted, workflows imported via the REST API, webhook calls, OpenRouter LLM provider): a "French" preference captured on the first message and **recalled on the second** — both via the native template (real LLM reply: "Your preferred language is French.") and via the community nodes; HTTP 400 rejection with no subject; `conversation.turn` events visible in `/v1/timeline`. The community-node test workflow is checked in: [`haki-e2e-test-workflow.json`](./haki-e2e-test-workflow.json).

## Honest limitations

A builder can break the chain (remove Haki Capture, wire the agent elsewhere) — n8n has no way to enforce the path. Coverage measurement (Context calls vs. Capture calls observed) is coming to the Haki console; today the rule is only guaranteed by the template and the nodes' own validation, not by interception.
