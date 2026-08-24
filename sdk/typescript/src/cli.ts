#!/usr/bin/env node
/** Haki CLI (TypeScript, zero framework — manual argv parsing).
 *
 *   haki-ts connect --api-url URL [--api-key KEY]   test /health, save ~/.haki/config.json
 *   haki-ts verify                                  end-to-end timed scenario, exit 0/1
 *   haki-ts status                                  API health
 *
 * The config file is the same as the Python CLI's: ~/.haki/config.json with
 * {"api_url", "api_key"} — both CLIs are interchangeable.
 */

import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { HakiClient, type PacketFact } from "./client.js";
import { HakiApiError, HakiError } from "./errors.js";

const CONFIG_PATH = join(homedir(), ".haki", "config.json");
const VERIFY_PROJECT = "prj_haki_verify";
const VERIFY_ORG = "org_haki_verify";

interface Config {
  api_url: string;
  api_key?: string | null;
}

function saveConfig(apiUrl: string, apiKey: string | null | undefined): void {
  mkdirSync(join(homedir(), ".haki"), { recursive: true });
  writeFileSync(
    CONFIG_PATH,
    JSON.stringify({ api_url: apiUrl, api_key: apiKey ?? null }, null, 2),
    "utf-8",
  );
}

function loadConfig(): Config {
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, "utf-8")) as Config;
  } catch {
    console.error(`no config at ${CONFIG_PATH} — run \`haki-ts connect --api-url URL\` first`);
    process.exit(1);
  }
}

function clientFromConfig(): HakiClient {
  const config = loadConfig();
  return new HakiClient({ baseUrl: config.api_url, apiKey: config.api_key ?? undefined });
}

function parseFlags(argv: string[]): { flags: Record<string, string>; ok: boolean } {
  const flags: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) return { flags, ok: false };
    const name = arg.slice(2);
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) return { flags, ok: false };
    flags[name] = value;
    i += 1;
  }
  return { flags, ok: true };
}

async function cmdConnect(argv: string[]): Promise<number> {
  const { flags, ok } = parseFlags(argv);
  if (!ok || !flags["api-url"]) {
    console.error("usage: haki-ts connect --api-url URL [--api-key KEY]");
    return 1;
  }
  const apiKey = flags["api-key"];
  const client = new HakiClient({ baseUrl: flags["api-url"], apiKey, timeout: 10_000 });
  const started = performance.now();
  try {
    const health = await client.health();
    const elapsed = performance.now() - started;
    saveConfig(flags["api-url"].replace(/\/+$/, ""), apiKey);
    // The API key is never printed back.
    console.log(
      `connected to ${flags["api-url"]} (${health.status}, db ${health.database}) in ${Math.round(elapsed)} ms`,
    );
    console.log(`config written to ${CONFIG_PATH}`);
    return 0;
  } catch (err) {
    if (err instanceof HakiError) console.error(`connect FAILED: ${err.message}`);
    else throw err;
    return 1;
  }
}

async function cmdStatus(): Promise<number> {
  try {
    const client = clientFromConfig();
    const started = performance.now();
    const health = await client.health();
    const elapsed = performance.now() - started;
    console.log(
      `api: ${health.status} | database: ${health.database} | health latency: ${Math.round(elapsed)} ms`,
    );
    return 0;
  } catch (err) {
    if (err instanceof HakiError) console.error(`status FAILED: ${err.message}`);
    else throw err;
    return 1;
  }
}

function step(label: string, started: number): void {
  const seconds = ((performance.now() - started) / 1000).toFixed(2);
  console.log(`  [${seconds.padStart(6)} s] ${label}`);
}

function factRecalled(facts: PacketFact[]): PacketFact | undefined {
  return facts.find((fact) => {
    const rendered = JSON.stringify(fact.value).toLowerCase();
    return rendered.includes("français") || rendered.includes("french") || rendered.includes('"fr"');
  });
}

/** Timed end-to-end scenario: capture -> consolidate -> new thread ->
 * context recalls the fact. Exit 0 on success, 1 on any failure.
 *
 * The captured event also carries a `mock_facts` mirror of the preference:
 * ignored by the real LLM extractor, it lets the scenario run end-to-end on
 * servers configured with HAKI_LLM_PROVIDER=fake (tests, offline dev).
 */
async function cmdVerify(): Promise<number> {
  const subjectId = `usr_verify_${randomUUID().replaceAll("-", "").slice(0, 12)}`;
  const thread1 = `thr_${randomUUID().replaceAll("-", "").slice(0, 8)}`;
  const thread2 = `thr_${randomUUID().replaceAll("-", "").slice(0, 8)}`; // new thread: memory must survive it

  console.log(`haki verify — subject ${subjectId}`);
  try {
    const config = loadConfig();
    if (!config.api_key) {
      try {
        const created = await new HakiClient({ baseUrl: config.api_url }).createKey({
          projectId: VERIFY_PROJECT,
          orgId: VERIFY_ORG,
          label: "haki verify bootstrap",
        });
        config.api_key = created.key;
        saveConfig(config.api_url, config.api_key);
        console.log(`  API key created for ${VERIFY_PROJECT} (saved to config)`);
      } catch (err) {
        if (!(err instanceof HakiError)) throw err;
        // dev-open mode, or key creation needs admin rights
      }
    }
    let client = clientFromConfig();
    const t0 = performance.now();

    try {
      await client.capture([
        {
          org_id: VERIFY_ORG,
          project_id: VERIFY_PROJECT,
          subject_type: "user",
          subject_id: subjectId,
          thread_id: thread1,
          kind: "conversation.message",
          occurred_at: new Date().toISOString(),
          payload: {
            role: "user",
            content: "Je préfère recevoir mes factures en français.",
            mock_facts: [
              {
                subject_id: subjectId,
                predicate: "invoice_language",
                value: { language: "fr" },
                qualifiers: {},
                confidence: 0.9,
                action: "create",
              },
            ],
          },
          idempotency_key: `verify-${randomUUID()}`,
        },
      ]);
    } catch (err) {
      // The configured credential (e.g. an admin key) cannot write data:
      // mint a project-scoped key with it, save it, retry once.
      if (!(err instanceof HakiApiError) || err.statusCode !== 401) throw err;
      const created = await client.createKey({
        projectId: VERIFY_PROJECT,
        orgId: VERIFY_ORG,
        label: "haki verify",
      });
      config.api_key = created.key;
      saveConfig(config.api_url, config.api_key);
      console.log(`  API key created for ${VERIFY_PROJECT} (saved to config)`);
      client = new HakiClient({ baseUrl: config.api_url, apiKey: created.key });
      await client.capture([
        {
          org_id: VERIFY_ORG,
          project_id: VERIFY_PROJECT,
          subject_type: "user",
          subject_id: subjectId,
          thread_id: thread1,
          kind: "conversation.message",
          occurred_at: new Date().toISOString(),
          payload: {
            role: "user",
            content: "Je préfère recevoir mes factures en français.",
            mock_facts: [
              {
                subject_id: subjectId,
                predicate: "invoice_language",
                value: { language: "fr" },
                qualifiers: {},
                confidence: 0.9,
                action: "create",
              },
            ],
          },
          idempotency_key: `verify-${randomUUID()}`,
        },
      ]);
    }
    step(`capture (thread ${thread1})`, t0);

    const t1 = performance.now();
    const result = await client.consolidateSubject({
      projectId: VERIFY_PROJECT,
      subjectId,
    });
    step(`consolidate: ${result.processed} job(s) processed`, t1);

    const t2 = performance.now();
    const response = await client.context({
      subjectId,
      query: "dans quelle langue envoyer les factures ?",
      projectId: VERIFY_PROJECT,
      purpose: `new thread ${thread2}`,
    });
    step(`context (new thread ${thread2})`, t2);

    const facts = response.packet.facts;
    const fact = factRecalled(facts);
    if (!fact) {
      console.log(`  trace_id: ${response.trace_id}`);
      console.log(
        `FAIL: preference not recalled (${facts.length} fact(s) served: ` +
          `${JSON.stringify(facts.map((f) => f.value))})`,
      );
      return 1;
    }
    console.log(`  recalled: ${fact.predicate} = ${JSON.stringify(fact.value)}`);
    console.log(`  trace_id: ${response.trace_id}`);
    console.log(`OK — total ${((performance.now() - t0) / 1000).toFixed(2)} s`);
    return 0;
  } catch (err) {
    if (err instanceof HakiError) console.error(`FAIL: ${err.message}`);
    else throw err;
    return 1;
  }
}

async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;
  switch (command) {
    case "connect":
      return cmdConnect(rest);
    case "verify":
      return cmdVerify();
    case "status":
      return cmdStatus();
    default:
      console.error(
        "usage: haki-ts <connect|verify|status>\n" +
          "  connect --api-url URL [--api-key KEY]   test /health, save ~/.haki/config.json\n" +
          "  verify                                  end-to-end timed scenario, exit 0/1\n" +
          "  status                                  API health",
      );
      return command === undefined || command === "help" || command === "--help" ? 0 : 1;
  }
}

// exitCode (not process.exit): let keep-alive sockets drain, otherwise
// Node on Windows can abort in libuv while closing handles at exit.
process.exitCode = await main(process.argv.slice(2));
