# gethaki

Python SDK for [Haki](https://gethaki.space) — a long-term memory layer for
AI agents. Every fact carries a date, a source and a status, so your agent
uses what's true today and can prove it.

- **Docs**: https://docs.gethaki.space
- **Source / self-hosting**: https://github.com/GetHaki/Haki
- **License**: Apache-2.0

## Install

```bash
pip install gethaki
```

## Usage

```python
from haki import HakiClient
from haki.runtime import build_prompt_context, capture_turn

client = HakiClient("http://localhost:8100")

# Before the LLM call: memory becomes an instruction block
packet = client.context(subject_id="usr_42", query=user_msg, project_id="prj")
prompt = build_prompt_context(packet) + "\n" + system_prompt

answer = my_llm(prompt, user_msg)  # your LLM call and app code stay unchanged

# After the LLM call: the conversation turn goes back into memory
capture_turn(client, "usr_42", "prj", user_msg, answer)
```

`AsyncHakiClient` is available for async codebases. Typed errors
(`HakiApiError`, `HakiConnectionError`) instead of bare exceptions.

## CLI

The `haki` command ships with this package:

```bash
haki connect --api-url http://localhost:8100
haki verify      # memorizes a fact, opens a new conversation, proves recall
haki status
haki mcp          # package this SDK for Cursor's MCP integration
```

## Requires

A running Haki API (self-hosted via `docker compose up -d` from the
[main repo](https://github.com/GetHaki/Haki), or a Haki Cloud project).
This package is the client only — see the repo for the server.
