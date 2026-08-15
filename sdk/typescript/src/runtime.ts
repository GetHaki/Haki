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
        "Cite the source when you rely on a fact. Facts already reflect the " +
        "CURRENT, resolved truth — an outdated value is removed the moment a " +
        "newer one is confirmed, so you never need to compare dates between " +
        "facts yourself; do not second-guess a fact's value. EXCEPTION: a fact " +
        "marked CONTESTED below is an unresolved disagreement — for those, and " +
        "only those, compare 'valid from' dates yourself and treat the most " +
        "recent one as current.",
    );
  }
  // A minimal terse rule under-performs a spelled-out chain of steps for
  // this exact task (Bug 3, 13 aout: gpt-4o-mini went 2/3 with a one-line
  // rule, 3/3 with these same three steps as a worked chain-of-note) --
  // only paid for the once-per-call header cost when a conflict is
  // actually being served, not on every ordinary packet.
  if (facts.some((f) => f.contested)) {
    lines.push(
      "One or more facts above are marked CONTESTED — an unresolved " +
        "disagreement between two dated values for the same real-world " +
        "fact, both shown so you are not left with zero information " +
        "instead of a wrong one. Resolve each contested group yourself: " +
        "1) find every CONTESTED fact that shares the same conflict id; " +
        "2) check whether the question has an EXPLICIT past-state " +
        "marker — 'before I changed/updated it', 'previously', 'used " +
        "to', 'originally', 'when I first started', 'in the first " +
        "[period]'. Ordinary past-tense phrasing alone ('what WAS X', " +
        "'how many did I have') is NOT this signal and still means the " +
        "CURRENT value — when in doubt, treat it as a CURRENT-value " +
        "question; 3) for a CURRENT-value question (the default), " +
        "treat ONLY the value with the LATEST date as current and " +
        "discard the earlier one entirely — do not mention it, do not " +
        "average it in, do not present both as still true; 4) only " +
        "when an explicit marker is present, answer with the EARLIER " +
        "dated value instead — defaulting to 'most recent' there " +
        "answers a different question than the one asked.",
    );
  }
  for (const fact of facts) {
    const value = JSON.stringify(fact.value);
    // Dual-date rendering (mechanism F1, 15 aout): an exact, precomputed
    // offset ("N days before the question") next to the ISO date, so the
    // reader VERIFIES a number instead of computing one from two ISO
    // dates -- a gpt-4o-mini-class reader gets that arithmetic right only
    // 13.5-16% of the time (Test-of-Time Arithmetic).
    let validFrom = fact.valid_from ?? "unknown date";
    if (fact.valid_from_relative) {
      validFrom = `${validFrom} — ${fact.valid_from_relative}`;
    }
    if (fact.temporal_range) {
      validFrom += `; described event dated ${fact.temporal_range.start} to ${fact.temporal_range.end}`;
    }
    const sources = (fact.source_event_ids ?? []).join(",") || "no-source";
    let marker = "";
    if (fact.freshness === "unconfirmed") {
      const last = fact.last_confirmed ?? "an unknown date";
      marker +=
        ` — UNCONFIRMED since ${last}: past its freshness horizon, ` +
        "re-confirm with the subject before relying on it";
    } else if (fact.freshness === "stale") {
      // 14 aout, mecanisme D (research/Diagnostic_Couverture_2026-08-14.md):
      // a fast-changing fact past its horizon is served, not hidden --
      // uncertain, not necessarily wrong.
      const last = fact.last_confirmed ?? "an unknown date";
      marker +=
        ` — STALE since ${last}: a fast-changing value past its freshness ` +
        "horizon, not necessarily wrong but not guaranteed current either " +
        "— treat it as the best available answer, not a certainty, and " +
        "prefer to re-confirm with the subject before relying on it for " +
        "anything consequential";
    }
    marker += fact.attributed_to
      ? ` [reported by a third party (${fact.attributed_to}) — not a statement by the subject]`
      : "";
    if (fact.contested) {
      marker +=
        ` — CONTESTED (conflict ${fact.conflict_id}): an unresolved conflicting ` +
        "value for this same fact is also shown below/above with the same " +
        "conflict id; use the one with the most recent 'valid from' date as " +
        "current, do not present both as equally true";
    }
    lines.push(
      `- ${fact.predicate}: ${value} (valid from ${validFrom}; sources: ${sources})${marker}`,
    );
  }
  if (episodes.length > 0) {
    lines.push(
      "Dated events from the source history (episodic memory): raw excerpts " +
        "kept for citation and narrative detail. They can mention values that " +
        "were later updated — if anything here conflicts with a fact above, " +
        "the FACT is the current, correct answer; never prefer an older " +
        "mention from here over it. If two dated items disagree and no fact " +
        "above covers it, use the one with the most recent date — UNLESS the " +
        "question explicitly asks about a past/previous state, in which case " +
        "use the one matching that earlier point in time instead.",
    );
    for (const episode of episodes) {
      let occurred = episode.occurred_at ?? "unknown date";
      if (episode.occurred_at_relative) {
        occurred = `${occurred} — ${episode.occurred_at_relative}`;
      }
      const marker = episode.context_neighbor
        ? " [surrounding context — not independently matched to the query, " +
          "included for the conversational moment around a result above]"
        : "";
      lines.push(
        `- [${occurred}] ${episode.kind}: ${episode.excerpt} (event: ${episode.event_id})${marker}`,
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
