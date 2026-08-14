"""Runtime helpers: the two agent hooks from the PRD (before/after the LLM).

- BEFORE the LLM call: `build_prompt_context(packet)` formats a ContextPacket
  into a delimited instruction block (facts with dates and sources) to
  prepend to the system prompt.
- AFTER the LLM call: `capture_turn(...)` writes the user/assistant exchange
  back to Haki as an event, so the consolidator can extract durable facts.

Usage (< 15 lines user-side):

    from haki import HakiClient
    from haki.runtime import build_prompt_context, capture_turn

    client = HakiClient("http://localhost:8100")
    packet = client.context(subject_id="usr_42", query=user_msg, project_id="prj")
    prompt = build_prompt_context(packet) + "\\n" + system_prompt
    answer = my_llm(prompt, user_msg)
    capture_turn(client, "usr_42", "prj", user_msg, answer)
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from haki.client import HakiClient


def build_prompt_context(packet: dict[str, Any]) -> str:
    """Format a ContextPacket as a delimited instruction block.

    Includes each fact's value, validity date and source event ids, plus the
    dated source events (episodic memory) so the agent can answer "what
    happened / when" questions and cite its memory. Empty packet -> empty
    string — UNLESS there is a `status` other than "ok" or a `warnings`
    entry to report: those must never be silently dropped just because no
    fact/episode happened to be packed (e.g. a total build_context failure
    surfaced as app.context.failed_packet has zero facts by construction,
    but the caller still needs to know memory could not be read, not read
    the empty block as "this subject has no memory").

    A packet emptied by the recall gate (empty_reason="no_relevant_memory",
    status "ok") renders as "" on purpose: injecting a "no relevant memory"
    block would itself be a distractor. The signal is for the CALLER
    (packet field), not for the prompt.
    """
    facts = (packet or {}).get("facts") or []
    episodes = (packet or {}).get("episodes") or []
    warnings = (packet or {}).get("warnings") or []
    status = (packet or {}).get("status", "ok")
    if not facts and not episodes and not warnings and status == "ok":
        return ""
    lines = ["<haki_memory>"]
    if status != "ok":
        lines.append(
            f"Memory status: {status}. Do not treat this as \"no memory\" — "
            "the retrieval itself was degraded or failed, memory may exist "
            "but could not be fully read."
        )
    if facts or episodes:
        lines.append(
            "Verified long-term memory facts about this subject. You MUST apply them "
            "whenever they are relevant to the request: treat them as instructions "
            "about HOW to respond (language of your answer, format, constraints, "
            "decisions already made), not as background trivia. If a fact states a "
            "language preference, write your entire response in that language. "
            "Cite the source when you rely on a fact. Facts already reflect the "
            "CURRENT, resolved truth — an outdated value is removed the moment a "
            "newer one is confirmed, so you never need to compare dates between "
            "facts yourself; do not second-guess a fact's value. EXCEPTION: a fact "
            "marked CONTESTED below is an unresolved disagreement — for those, and "
            "only those, compare 'valid from' dates yourself and treat the most "
            "recent one as current.",
        )
    # A minimal terse rule under-performs a spelled-out chain of steps for
    # this exact task (Bug 3, 13 aout: gpt-4o-mini went 2/3 with a one-line
    # rule, 3/3 with these same three steps as a worked chain-of-note) --
    # only paid for the once-per-call header cost when a conflict is
    # actually being served, not on every ordinary packet.
    if any(f.get("contested") for f in facts):
        lines.append(
            "One or more facts above are marked CONTESTED — an unresolved "
            "disagreement between two dated values for the same real-world "
            "fact, both shown so you are not left with zero information "
            "instead of a wrong one. Resolve each contested group yourself: "
            "1) find every CONTESTED fact that shares the same conflict id; "
            "2) check whether the question has an EXPLICIT past-state "
            "marker — 'before I changed/updated it', 'previously', 'used "
            "to', 'originally', 'when I first started', 'in the first "
            "[period]'. Ordinary past-tense phrasing alone ('what WAS X', "
            "'how many did I have') is NOT this signal and still means the "
            "CURRENT value — when in doubt, treat it as a CURRENT-value "
            "question; 3) for a CURRENT-value question (the default), "
            "treat ONLY the value with the LATEST date as current and "
            "discard the earlier one entirely — do not mention it, do not "
            "average it in, do not present both as still true; 4) only "
            "when an explicit marker is present, answer with the EARLIER "
            "dated value instead — defaulting to 'most recent' there "
            "answers a different question than the one asked."
        )
    for fact in facts:
        value = fact.get("value")
        valid_from = fact.get("valid_from") or "unknown date"
        sources = ",".join(fact.get("source_event_ids") or []) or "no-source"
        marker = ""
        if fact.get("freshness") == "unconfirmed":
            last = fact.get("last_confirmed") or "an unknown date"
            marker = (
                f" — UNCONFIRMED since {last}: past its freshness horizon, "
                "re-confirm with the subject before relying on it"
            )
        elif fact.get("freshness") == "stale":
            last = fact.get("last_confirmed") or "an unknown date"
            marker = (
                f" — STALE since {last}: a fast-changing value past its "
                "freshness horizon, not necessarily wrong but not "
                "guaranteed current either — treat it as the best available "
                "answer, not a certainty, and prefer to re-confirm with the "
                "subject before relying on it for anything consequential"
            )
        if fact.get("attributed_to"):
            marker += (
                f" [reported by a third party ({fact['attributed_to']}) — "
                "not a statement by the subject]"
            )
        if fact.get("contested"):
            marker += (
                " — CONTESTED (conflict "
                f"{fact.get('conflict_id')}): an unresolved conflicting value for "
                "this same fact is also shown below/above with the same conflict "
                "id; use the one with the most recent 'valid from' date as current, "
                "do not present both as equally true"
            )
        lines.append(
            f"- {fact.get('predicate')}: {value} "
            f"(valid from {valid_from}; sources: {sources}){marker}"
        )
    if episodes:
        lines.append(
            "Dated events from the source history (episodic memory): raw excerpts "
            "kept for citation and narrative detail. They can mention values that "
            "were later updated — if anything here conflicts with a fact above, "
            "the FACT is the current, correct answer; never prefer an older "
            "mention from here over it. If two dated items disagree and no fact "
            "above covers it, use the one with the most recent date — UNLESS the "
            "question explicitly asks about a past/previous state, in which case "
            "use the one matching that earlier point in time instead."
        )
        for episode in episodes:
            occurred = episode.get("occurred_at") or "unknown date"
            lines.append(
                f"- [{occurred}] {episode.get('kind')}: {episode.get('excerpt')} "
                f"(event: {episode.get('event_id')})"
            )
    for warning in warnings:
        lines.append(f"! {warning}")
    lines.append("</haki_memory>")
    return "\n".join(lines)


def capture_turn(
    client: HakiClient,
    subject_id: str,
    project_id: str,
    user_msg: str,
    assistant_msg: str,
    *,
    org_id: str = "org_default",
    agent_id: str | None = None,
    thread_id: str | None = None,
    kind: str = "conversation.turn",
) -> dict[str, Any]:
    """After-LLM hook: capture one user/assistant exchange as a Haki event."""
    return client.capture(
        [
            {
                "org_id": org_id,
                "project_id": project_id,
                "subject_type": "user",
                "subject_id": subject_id,
                "agent_id": agent_id,
                "thread_id": thread_id,
                "kind": kind,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                },
                "idempotency_key": f"turn-{uuid.uuid4()}",
            }
        ]
    )
