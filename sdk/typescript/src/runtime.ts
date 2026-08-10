/** Runtime helpers: the two agent hooks from the PRD (before/after the LLM).
 *
 * - BEFORE the LLM call: `buildPromptContext(packet)` formats a ContextPacket
 *   into a delimited instruction block (facts with dates and sources) to
 *   prepend to the system prompt.
 * - AFTER the LLM call: `captureTurn(...)` writes the user/assistant exchange
 *   back to Haki as an event, so the consolidator can extract durable facts.
 *
 * Usage (< 15 lines user-side):
 *
 *   import { HakiClient } from "gethaki";
 *   import { buildPromptContext, captureTurn } from "gethaki/runtime";
 *
 *   const client = new HakiClient({ baseUrl: "http://localhost:8100", apiKey });
 *   const { packet } = await client.context({ subjectId: "usr_42", query: userMsg, projectId: "prj" });
 *   const prompt = buildPromptContext(packet) + "\n" + systemPrompt;
 *   const answer = await myLlm(prompt, userMsg);
 *   await captureTurn(client, { subjectId: "usr_42", projectId: "prj", userMsg, assistantMsg: answer });
 */

import { randomUUID } from "node:crypto";

import type { CaptureResponse, ContextPacket, HakiClient } from "./client.js";

/** Format a ContextPacket as a delimited instruction block.
 *
 * Includes each fact's value, validity date and source event ids, plus the
 * dated source events (episodic memory), so the agent can trust and cite
 * its memory — parity with the Python SDK's version. Empty packet -> empty
 * string — UNLESS there is a `status` other than "ok" or a `warnings` entry
 * to report: those must never be silently dropped just because no
 * fact/episode happened to be packed (e.g. a total build_context failure
 * has zero facts by construction, but the caller still needs to know
 * memory could not be read, not read the empty block as "no memory").
 *
 * A packet emptied by the recall gate (emptyReason "no_relevant_memory",
 * status "ok") renders as "" on purpose: injecting a "no relevant memory"
 * block would itself be a distractor. The signal is for the CALLER
 * (packet field), not for the prompt.
 */
export function buildPromptContext(packet: ContextPacket | null | undefined): string {
  const facts = packet?.facts ?? [];
  const episodes = packet?.episodes ?? [];
  const warnings = packet?.warnings ?? [];
  const status = packet?.status ?? "ok";
  if (facts.length === 0 && episodes.length === 0 && warnings.length === 0 && status === "ok") {
    return "";
  }
  const lines = ["<haki_memory>"];
  if (status !== "ok") {
    lines.push(
      `Memory status: ${status}. Do not treat this as "no memory" — the ` +
        "retrieval itself was degraded or failed, memory may exist but " +
        "could not be fully read.",
    );
  }
  if (facts.length > 0 || episodes.length > 0) {
    lines.push(
      "Verified long-term memory facts about this subject. You MUST apply them " +
        "whenever they are relevant to the request: treat them as instructions " +
        "about HOW to respond (language of your answer, format, constraints, " +
        "decisions already made), not as background trivia. If a fact states a " +
        "language preference, write your entire response in that language. " +
        "Cite the source when you rely on a fact.",
    );
  }
  for (const fact of facts) {
    const value = JSON.stringify(fact.value);
    const validFrom = fact.valid_from ?? "unknown date";
    const sources = (fact.source_event_ids ?? []).join(",") || "no-source";
    lines.push(`- ${fact.predicate}: ${value} (valid from ${validFrom}; sources: ${sources})`);
  }
  if (episodes.length > 0) {
    lines.push("Dated events from the source history (episodic memory):");
    for (const episode of episodes) {
      const occurred = episode.occurred_at ?? "unknown date";
      lines.push(
        `- [${occurred}] ${episode.kind}: ${episode.excerpt} (event: ${episode.event_id})`,
      );
    }
  }
  for (const warning of warnings) {
    lines.push(`! ${warning}`);
  }
  lines.push("</haki_memory>");
  return lines.join("\n");
}

/** After-LLM hook: capture one user/assistant exchange as a Haki event. */
export function captureTurn(
  client: HakiClient,
  turn: {
    subjectId: string;
    projectId: string;
    userMsg: string;
    assistantMsg: string;
    orgId?: string;
    agentId?: string;
    threadId?: string;
    kind?: string;
  },
): Promise<CaptureResponse> {
  return client.capture([
    {
      org_id: turn.orgId ?? "org_default",
      project_id: turn.projectId,
      subject_type: "user",
      subject_id: turn.subjectId,
      agent_id: turn.agentId,
      thread_id: turn.threadId,
      kind: turn.kind ?? "conversation.turn",
      occurred_at: new Date().toISOString(),
      payload: {
        messages: [
          { role: "user", content: turn.userMsg },
          { role: "assistant", content: turn.assistantMsg },
        ],
      },
      idempotency_key: `turn-${randomUUID()}`,
    },
  ]);
}
