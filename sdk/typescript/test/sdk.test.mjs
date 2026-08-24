/** Haki TypeScript SDK tests (node:test), against the REAL API.
 *
 * Prerequisites: a Haki API running on HAKI_TEST_API_URL (default
 * http://localhost:8100) with HAKI_AUTH_REQUIRED=true and
 * HAKI_LLM_PROVIDER=fake / HAKI_EMBED_PROVIDER=fake.
 *
 * Test keys are created via HAKI_TEST_ADMIN_KEY when the server sets
 * HAKI_ADMIN_KEY; otherwise the documented bootstrap (first key free on an
 * empty api_keys table) is used.
 */

import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  HakiApiError,
  HakiClient,
  buildPromptContext,
  captureTurn,
} from "../dist/index.js";

const API_URL = process.env.HAKI_TEST_API_URL ?? "http://localhost:8100";
const ADMIN_KEY = process.env.HAKI_TEST_ADMIN_KEY;
const SDK_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");

// Unique project ids per run: tests never interfere with each other's data.
const RUN = randomUUID().slice(0, 8);
const PROJECT = `prj_ts_${RUN}`;
const ORG = `org_ts_${RUN}`;

/** Create an API key for a project (admin key if configured, else bootstrap). */
async function makeKey(projectId, orgId) {
  const client = new HakiClient({ baseUrl: API_URL, apiKey: ADMIN_KEY });
  const created = await client.createKey({ projectId, orgId, label: "ts test" });
  return created.key;
}

function mockFact(predicate, value, subjectId) {
  return {
    subject_id: subjectId,
    predicate,
    value,
    qualifiers: {},
    confidence: 0.9,
    action: "create",
  };
}

function memoryEvent(subjectId, facts, projectId = PROJECT) {
  return {
    org_id: ORG,
    project_id: projectId,
    subject_type: "user",
    subject_id: subjectId,
    kind: "conversation.message",
    occurred_at: new Date().toISOString(),
    payload: { role: "user", content: "...", mock_facts: facts },
    idempotency_key: `ts-${randomUUID()}`,
  };
}

test("capture -> consolidate -> context round-trip (fake provider)", async () => {
  const key = await makeKey(PROJECT, ORG);
  const client = new HakiClient({ baseUrl: API_URL, apiKey: key });
  const subjectId = `usr_${RUN}`;

  const body = await client.capture(
    [memoryEvent(subjectId, [mockFact("invoice_language", { language: "fr" }, subjectId)])],
    `batch-${randomUUID()}`,
  );
  assert.equal(body.status, "accepted");
  assert.ok(body.consolidation_job_id);

  const result = await client.consolidateSubject({ projectId: PROJECT, subjectId });
  assert.ok(result.processed >= 1);

  // New thread: the memory must survive the thread boundary.
  const response = await client.context({
    subjectId,
    query: "invoice_language",
    projectId: PROJECT,
    purpose: "new thread",
  });
  assert.deepEqual(
    response.packet.facts.map((f) => f.value),
    [{ language: "fr" }],
  );
  assert.ok(response.token_count > 0);

  const trace = await client.inspect(response.trace_id, {
    projectId: PROJECT,
    subjectId,
  });
  assert.equal(trace.query, "invoice_language");
  assert.equal(trace.decisions[0].action, "included");

  const timeline = await client.timeline({ projectId: PROJECT, subjectId });
  assert.equal(timeline.events.length, 1);
});

test("buildPromptContext: value, date, source and hardened instruction", () => {
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "invoice_language",
        value: { language: "fr" },
        confidence: 0.9,
        valid_from: "2026-07-28T10:00:00+00:00",
        source_event_ids: ["evt-123"],
      },
    ],
    warnings: ["open_conflict: 1 fact(s) hidden pending conflict resolution"],
  };
  const block = buildPromptContext(packet);
  assert.ok(block.includes("<haki_memory>") && block.includes("</haki_memory>"));
  assert.ok(block.includes("invoice_language"));
  assert.ok(block.includes("fr"));
  assert.ok(block.includes("2026-07-28"));
  assert.ok(block.includes("evt-123"));
  assert.ok(block.includes("open_conflict"));
  assert.ok(block.includes("You MUST apply them"));
  assert.ok(block.includes("write your entire response in that language"));
  // Empty packet -> empty block, safe to prepend.
  assert.equal(buildPromptContext({ facts: [], warnings: [] }), "");
  assert.equal(buildPromptContext(null), "");
  // M3 recall gate: a gate-emptied packet (status ok, empty_reason set)
  // also renders as "" -- no "no relevant memory" distractor block.
  assert.equal(
    buildPromptContext({
      facts: [],
      episodes: [],
      warnings: [],
      status: "ok",
      empty_reason: "no_relevant_memory",
    }),
    "",
  );
});

test("buildPromptContext: unconfirmed and stale freshness markers (14 aout, mecanisme D)", () => {
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "employer",
        value: { name: "Dicken AI" },
        confidence: 0.9,
        valid_from: "2024-01-01T00:00:00+00:00",
        source_event_ids: [],
        freshness: "unconfirmed",
        last_confirmed: "2024-01-01T00:00:00+00:00",
      },
      {
        id: "f2",
        predicate: "current_project",
        value: { name: "Atlas" },
        confidence: 0.9,
        valid_from: "2026-08-01T00:00:00+00:00",
        source_event_ids: [],
        freshness: "stale",
        last_confirmed: "2026-08-01T00:00:00+00:00",
      },
      {
        id: "f3",
        predicate: "invoice_language",
        value: { language: "fr" },
        confidence: 0.9,
        valid_from: "2026-08-01T00:00:00+00:00",
        source_event_ids: [],
        freshness: "current",
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(packet);
  assert.ok(block.includes("UNCONFIRMED since 2024-01-01"));
  assert.ok(block.includes("STALE since 2026-08-01"));
  assert.ok(block.includes("re-confirm with the subject"));
  // A plain "current" fact carries neither marker.
  const currentLine = block.split("\n").find((l) => l.includes("invoice_language"));
  assert.ok(!currentLine.includes("UNCONFIRMED") && !currentLine.includes("STALE"));
});

test("buildPromptContext: facts outrank episodes on conflict (Bug 3, 13 aout)", () => {
  // Once episodes carry raw historical text alongside facts (key merging,
  // 13 aout), an episode can mention a value since superseded. The prompt
  // must say a fact is the already-resolved current truth and wins over a
  // conflicting episode mention -- the guard the 11 aout oracle test showed
  // gpt-4o-mini needs spelled out, not left implicit. Parity with the
  // Python SDK's equivalent test.
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "wells_fargo_pre_approval",
        value: { amount: "$400,000" },
        valid_from: "2023-11-30T00:00:00+00:00",
        source_event_ids: ["evt-2"],
      },
    ],
    episodes: [
      {
        event_id: "evt-1",
        kind: "conversation.turn",
        occurred_at: "2023-08-11T00:00:00+00:00",
        excerpt: "user: I got pre-approved for $350,000 from Wells Fargo.",
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(packet);
  assert.ok(block.includes("the FACT is the current, correct answer"));
  assert.ok(block.includes("CURRENT, resolved truth"));
});

test("buildPromptContext: marks context window neighbors (F2, 15 aout)", () => {
  // A `context_neighbor` episode (added for surrounding context, not
  // independently matched to the query) is marked as such in the
  // rendered prompt. Parity with the Python SDK's equivalent test.
  const packet = {
    facts: [],
    episodes: [
      {
        event_id: "evt-1",
        kind: "chat_session",
        occurred_at: "2023-05-07T12:00:00+00:00",
        excerpt: "user: Zolgorvex mentioned a favorite pastime.",
        context_neighbor: false,
      },
      {
        event_id: "evt-2",
        kind: "chat_session",
        occurred_at: "2023-05-07T13:00:00+00:00",
        excerpt: "user: Unrelated later chat.",
        context_neighbor: true,
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(packet);
  const ordinaryLine = block.split("\n").find((line) => line.includes("evt-1"));
  const neighborLine = block.split("\n").find((line) => line.includes("evt-2"));
  assert.ok(!ordinaryLine.includes("surrounding context"));
  assert.ok(neighborLine.includes("surrounding context"));
});

test("buildPromptContext: renders dual dates and temporal_range (F1, 15 aout)", () => {
  // Every dated packet item carries its ISO date AND an exact,
  // precomputed offset from the question ("N days before the question").
  // Parity with the Python SDK's equivalent test.
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "hiking_trip",
        value: { trail: "Congress Trail" },
        valid_from: "2023-06-04T00:00:00+00:00",
        valid_from_relative: "21 days (3 weeks) before the question",
        temporal_range: { start: "2023-06-18", end: "2023-06-25" },
        source_event_ids: ["evt-1"],
      },
    ],
    episodes: [
      {
        event_id: "evt-2",
        kind: "chat_session",
        occurred_at: "2023-06-04T00:00:00+00:00",
        occurred_at_relative: "21 days (3 weeks) before the question",
        excerpt: "user: went hiking last week",
        context_neighbor: false,
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(packet);
  assert.ok(block.includes("21 days (3 weeks) before the question"));
  assert.ok(block.includes("2023-06-18 to 2023-06-25"));
});

test("buildPromptContext: contested facts are marked, with the tie-break exception (13 aout)", () => {
  // "Stop hiding real conflicts": a genuine two-sided disagreement is now
  // served (app.context), both facts sharing a conflict_id, instead of an
  // empty packet. Parity with the Python SDK's equivalent test.
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "language",
        value: { lang: "fr" },
        valid_from: "2026-07-28T10:00:00+00:00",
        source_event_ids: ["evt-1"],
        contested: true,
        conflict_id: "c1",
      },
      {
        id: "f2",
        predicate: "language",
        value: { lang: "en" },
        valid_from: "2026-07-29T10:00:00+00:00",
        source_event_ids: ["evt-2"],
        contested: true,
        conflict_id: "c1",
      },
    ],
    warnings: ["open_conflict: 2 fact(s) served with an unresolved conflicting value"],
  };
  const block = buildPromptContext(packet);
  // Header exception sentence (1) + dedicated chain-of-note paragraph
  // (2, only emitted when a fact is actually contested) + once per
  // contested fact (2) -- five total.
  assert.equal((block.match(/CONTESTED/g) ?? []).length, 5);
  assert.ok(block.includes("find every CONTESTED fact that shares the same conflict id"));
  assert.ok(block.includes("conflict c1"));
  assert.ok(block.includes("compare 'valid from' dates yourself"));
  // An ordinary (non-contested) fact never gets the per-fact marker (the
  // header sentence always explains the exception, so "CONTESTED" alone
  // is not a useful signal here -- the marker phrase is).
  const ordinary = buildPromptContext({
    facts: [
      {
        id: "f3",
        predicate: "plan",
        value: { tier: "pro" },
        valid_from: "2026-07-28T10:00:00+00:00",
        source_event_ids: ["evt-3"],
      },
    ],
    warnings: [],
  });
  assert.ok(!ordinary.includes("— CONTESTED (conflict"));
});

test("buildPromptContext: flags auto-reclassified facts (16 aout, reclassification safety net)", () => {
  // Parity with the Python SDK's equivalent test: a fact activated by the
  // automatic overflow reclassification (mechanism C) is served like any
  // other, but flagged so the reader can judge whether 3 "occurrences"
  // actually look like updates to one attribute instead.
  const packet = {
    facts: [
      {
        id: "f1",
        predicate: "office_city",
        value: { city: "Dakar" },
        valid_from: "2026-07-28T10:00:00+00:00",
        source_event_ids: ["evt-1"],
        auto_reclassified: true,
      },
      {
        id: "f2",
        predicate: "plan",
        value: { tier: "pro" },
        valid_from: "2026-07-28T10:00:00+00:00",
        source_event_ids: ["evt-2"],
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(packet);
  assert.equal((block.match(/AUTO-RECLASSIFIED/g) ?? []).length, 1);
  const lines = block.split("\n");
  assert.ok(lines.some((l) => l.includes("office_city") && l.includes("AUTO-RECLASSIFIED")));
  assert.ok(!lines.some((l) => l.includes("plan:") && l.includes("AUTO-RECLASSIFIED")));
});

test("buildPromptContext: carries a past-value exception (13 aout, LongMemEval)", () => {
  // The "always prefer most recent" guard is wrong when the question
  // itself asks about a PAST value (real case: Apex Legends level goal,
  // qid 9bbe84a2, run 31705865474) -- parity with the Python SDK test.
  const contestedPacket = {
    facts: [
      {
        id: "f1",
        predicate: "apex_legends_level_goal",
        value: { goal: 100 },
        valid_from: "2023-06-16T00:00:00+00:00",
        source_event_ids: ["evt-1"],
        contested: true,
        conflict_id: "c1",
      },
      {
        id: "f2",
        predicate: "apex_legends_level_goal",
        value: { goal: 150 },
        valid_from: "2023-09-30T00:00:00+00:00",
        source_event_ids: ["evt-2"],
        contested: true,
        conflict_id: "c1",
      },
    ],
    warnings: [],
  };
  const block = buildPromptContext(contestedPacket);
  assert.ok(block.includes("EXPLICIT past-state marker"));
  assert.ok(block.includes("Ordinary past-tense phrasing alone"));
  assert.ok(block.includes("answer with the EARLIER dated value instead"));

  const episodesPacket = {
    facts: [],
    episodes: [
      {
        event_id: "evt-1",
        kind: "conversation.turn",
        occurred_at: "2023-06-16T00:00:00+00:00",
        excerpt: "user: my Apex Legends goal is level 100.",
      },
    ],
    warnings: [],
  };
  const block2 = buildPromptContext(episodesPacket);
  assert.ok(block2.includes("UNLESS the question explicitly asks about a past/previous state"));
});

test("no API key -> HakiApiError 401 unauthorized", async () => {
  const client = new HakiClient({ baseUrl: API_URL });
  await assert.rejects(
    client.context({ subjectId: "usr_x", query: "q", projectId: PROJECT }),
    (err) => {
      assert.ok(err instanceof HakiApiError);
      assert.equal(err.statusCode, 401);
      assert.equal(err.errorType, "unauthorized");
      assert.equal(err.field, "Authorization");
      assert.ok(err.message.includes("[401 unauthorized]"));
      return true;
    },
  );
});

test("key of project A used on project B -> 403 forbidden_scope", async () => {
  const keyA = await makeKey(`prj_a_${RUN}`, ORG);
  const client = new HakiClient({ baseUrl: API_URL, apiKey: keyA });
  await assert.rejects(
    client.context({ subjectId: "usr_x", query: "q", projectId: `prj_b_${RUN}` }),
    (err) => {
      assert.ok(err instanceof HakiApiError);
      assert.equal(err.statusCode, 403);
      assert.equal(err.errorType, "forbidden_scope");
      return true;
    },
  );
});

test("forget subject -> context serves nothing anymore", async () => {
  const key = await makeKey(PROJECT, ORG);
  const client = new HakiClient({ baseUrl: API_URL, apiKey: key });
  const subjectId = `usr_forget_${RUN}`;

  await client.capture([
    memoryEvent(subjectId, [mockFact("plan", { tier: "pro" }, subjectId)]),
  ]);
  await client.consolidateSubject({ projectId: PROJECT, subjectId });
  const before = await client.context({ subjectId, query: "plan", projectId: PROJECT });
  assert.equal(before.packet.facts.length, 1);

  const receipt = await client.forget({ projectId: PROJECT, subjectId, mode: "disable" });
  assert.equal(receipt.scope, "subject");
  assert.ok(receipt.facts_disabled >= 1);

  const after = await client.context({ subjectId, query: "plan", projectId: PROJECT });
  assert.equal(after.packet.facts.length, 0);
});

test("captureTurn produces an event visible in the timeline", async () => {
  const key = await makeKey(PROJECT, ORG);
  const client = new HakiClient({ baseUrl: API_URL, apiKey: key });
  const subjectId = `usr_turn_${RUN}`;
  const threadId = `thr_${RUN}`;

  await captureTurn(client, {
    subjectId,
    projectId: PROJECT,
    userMsg: "Bonjour",
    assistantMsg: "Bonjour !",
    threadId,
  });

  const timeline = await client.timeline({ projectId: PROJECT, subjectId });
  assert.equal(timeline.events.length, 1);
  const event = timeline.events[0];
  assert.equal(event.kind, "conversation.turn");
  assert.equal(event.thread_id, threadId);
  assert.deepEqual(event.payload.messages, [
    { role: "user", content: "Bonjour" },
    { role: "assistant", content: "Bonjour !" },
  ]);
  assert.ok(event.idempotency_key.startsWith("turn-"));
});

test("CLI: connect writes a Python-compatible config, verify exits 0", async () => {
  // The verify scenario is bound to the fixed project prj_haki_verify.
  const key = await makeKey("prj_haki_verify", "org_haki_verify");

  // Isolate the config: the CLI must not touch the real ~/.haki/config.json.
  const fakeHome = mkdtempSync(join(tmpdir(), "haki-ts-test-"));
  const env = { ...process.env, HOME: fakeHome, USERPROFILE: fakeHome };

  const connect = spawnSync(
    process.execPath,
    [join(SDK_DIR, "dist", "cli.js"), "connect", "--api-url", API_URL, "--api-key", key],
    { env, encoding: "utf-8" },
  );
  assert.equal(connect.status, 0, connect.stderr || connect.stdout);
  assert.ok(connect.stdout.includes("connected to"));

  // Same file and format as the Python CLI.
  const config = JSON.parse(
    readFileSync(join(fakeHome, ".haki", "config.json"), "utf-8"),
  );
  assert.equal(config.api_url, API_URL.replace(/\/+$/, ""));
  assert.equal(config.api_key, key);

  const verify = spawnSync(
    process.execPath,
    [join(SDK_DIR, "dist", "cli.js"), "verify"],
    { env, encoding: "utf-8", timeout: 120_000 },
  );
  assert.equal(verify.status, 0, verify.stderr || verify.stdout);
  assert.ok(verify.stdout.includes("recalled: invoice_language"));
  assert.ok(verify.stdout.includes("OK — total"));
});
