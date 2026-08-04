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
            "Cite the source when you rely on a fact.",
        )
    for fact in facts:
        value = fact.get("value")
        valid_from = fact.get("valid_from") or "unknown date"
        sources = ",".join(fact.get("source_event_ids") or []) or "no-source"
        lines.append(
            f"- {fact.get('predicate')}: {value} "
            f"(valid from {valid_from}; sources: {sources})"
        )
    if episodes:
        lines.append("Dated events from the source history (episodic memory):")
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
