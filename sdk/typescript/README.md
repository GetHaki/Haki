# gethaki

TypeScript SDK for [Haki](https://gethaki.space) — a long-term memory layer
for AI agents. Every fact carries a date, a source and a status, so your
agent uses what's true today and can prove it.

Parity with the Python SDK: same methods, same typed errors, the same
`<haki_memory>` prompt block — zero runtime dependency (native `fetch`,
Node ≥ 18).

- **Docs**: https://docs.gethaki.space
- **Source / self-hosting**: https://github.com/GetHaki/Haki
- **License**: Apache-2.0

## Install

```bash
npm install gethaki
```

## Usage

```typescript
import { HakiClient, buildPromptContext, captureTurn } from "gethaki";

const client = new HakiClient({ baseUrl: "http://localhost:8100", apiKey: "hk_..." });

const { packet } = await client.context({ subjectId: "usr_42", query: userMsg, projectId: "prj" });
const prompt = buildPromptContext(packet) + "\n" + systemPrompt;
const answer = await myLlm(prompt, userMsg);
await captureTurn(client, { subjectId: "usr_42", projectId: "prj", userMsg, assistantMsg: answer });
```

## CLI

```bash
npx haki-ts connect --api-url http://localhost:8100
npx haki-ts verify
npx haki-ts status
```

## Requires

A running Haki API (self-hosted via `docker compose up -d` from the
[main repo](https://github.com/GetHaki/Haki), or a Haki Cloud project).
This package is the client only — see the repo for the server.
