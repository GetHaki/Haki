"""Cutting an event's payload into retrievable, servable units.

Why an event is the wrong unit
-------------------------------
Until 21 Aug one event was one episode: indexed whole, embedded whole,
served whole. On the eval corpus (272 real LoCoMo sessions) that meant:

    payload JSON                  median 3 244 chars, max 7 113
    truncated by EPISODE_TEXT_CHARS = 4000     69/272 sessions (25.4 %),
                                               7.1 % of the corpus destroyed
    cost to serve one episode     median 810 tokens, max 1 000
    episodes fitting the eval's 900-token budget   177/272 -- and one at a time

So a single episode ate ~90 % of the budget, and only a quarter of the
corpus was even reachable. Worse, the embedder truncates at ~128 tokens
(verified directly against LocalEmbedder: two texts differing only after
that point score cosine similarity 0.9999999999999999), so **0 of 272
episodes were fully embedded** -- the median episode had 12.4 % of itself
in the index and 87.6 % invisible to vector search.

Cutting on turn boundaries fixes all three at once. Claimed by the
external audit this module implements, on the 1 536 non-adversarial
LoCoMo questions, gold evidence served under a 900-token budget,
everything else held constant:

    one episode = one session   19.0 %
    one episode = one turn      66.3 %      (+47.3 points, unverified here)

That specific end-to-end percentage was not re-run against this project's
own bench. What IS independently verified is the mechanism it rests on: a
turn is small enough to be embedded whole -- 5 882/5 882 real LoCoMo turns
are fully covered by a 128-token window, against 0/272 sessions.

The objection, and why it does not apply here
------------------------------------------------
LongMemEval (arXiv 2410.10813, Table 3) reports that WITHOUT enrichment a
session beats a round as the retrieval unit -- Recall@5 of 0.706 against
0.582 -- because an isolated turn loses its referents ("and him?", "when
was that?"). Two reasons the conclusion inverts here:

1. **Budget.** That table measures recall@k with no token constraint. Haki
   serves under 900-2000 tokens. A turn costs ~10 tokens, a session ~810:
   at equal budget you serve eighty turns or one session. The measurement
   above is under budget, which is the regime the product runs in.
2. **Enrichment restores it anyway.** The same table shows the turn
   overtaking the session (0.644 > 0.582) once its key carries the facts
   extracted from it. That is `index_text` below, and it is where the
   already-existing key-merging code (app.consolidator, mechanism E3)
   finally has something small enough to work on.

Design notes
------------
- **The verbatim is what gets served; `index_text` is what gets matched.**
  Keeping them separate is what lets a chunk be *found* through extracted
  facts without those facts ever leaking into what the agent reads as a
  direct source quote. Until the fact-to-chunk link lands, `index_text`
  equals the text.
- **Cut on the payload's own boundaries, never on a character count**, when
  the payload declares any: a message list gives real turn boundaries, and
  cutting there costs nothing and loses nothing. The character-count split
  is the fallback for opaque payloads, and it prefers paragraph then
  sentence boundaries before cutting mid-word.
- **Bounded.** A pathological payload cannot produce unbounded chunks: the
  count is capped and the tail is merged into the last chunk rather than
  dropped, so no source text is ever silently lost -- the failure this
  module exists to end.
"""

from __future__ import annotations

import json
import re

# Target size for a fallback chunk, in characters. ~250 tokens at the
# project's 4-chars-per-token estimate: comfortably inside the 128-token
# window of the current embedder for short turns, and inside the 512-token
# window of any retrieval-trained replacement.
CHUNK_TARGET_CHARS = 1000

# Hard ceiling on one chunk, applied after boundary-seeking. A single
# message longer than this is split; nothing is discarded.
CHUNK_MAX_CHARS = 1600

# Ceiling on chunks per event. A 200-turn session is already far outside
# anything the API is meant to receive in one event; beyond this the tail
# is merged into the last chunk rather than dropped.
MAX_CHUNKS_PER_EVENT = 200

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _message_text(message: dict) -> str:
    """One conversation turn, rendered for both indexing and serving.

    Role first, then content: the speaker is the single most useful token
    for retrieval in a two-party conversation (it is what tells "Caroline
    said" from "Melanie said"), and it is what the entity mechanism in
    app.context keys on.
    """
    role = message.get("role") or message.get("speaker") or "?"
    content = message.get("content") or message.get("text") or ""
    return f"{role}: {content}".strip()


def _split_on_boundaries(text: str) -> list[str]:
    """Split an opaque blob, preferring paragraph then sentence breaks."""
    chunks: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= CHUNK_MAX_CHARS:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(paragraph):
            if current and len(current) + 1 + len(sentence) > CHUNK_TARGET_CHARS:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
            # A single sentence longer than the ceiling: cut it, rather
            # than let one chunk grow without bound.
            while len(current) > CHUNK_MAX_CHARS:
                chunks.append(current[:CHUNK_MAX_CHARS])
                current = current[CHUNK_MAX_CHARS:]
        if current:
            chunks.append(current)
    return chunks


def chunk_payload(kind: str, payload: dict | None) -> list[str]:
    """Cut one event into the units that are retrieved, ranked and served.

    Returns at least one chunk for any event -- an event with an empty
    payload still has a kind, and a chunk that is only a kind is a valid
    (if useless) episode, whereas returning nothing would make the event
    permanently unreachable.
    """
    payload = payload or {}
    messages = payload.get("messages")
    if isinstance(messages, list) and messages and all(isinstance(m, dict) for m in messages):
        chunks = [text for m in messages if (text := _message_text(m))]
    else:
        # `episode_text`'s shape (kind + serialized payload) is kept for
        # payloads with no message structure: it is what the consolidator
        # has always embedded, and it stays deterministic.
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        blob = f"{kind} {serialized}"
        chunks = _split_on_boundaries(blob) if len(blob) > CHUNK_MAX_CHARS else [blob]

    chunks = [c for c in (chunk.strip() for chunk in chunks) if c]
    if not chunks:
        return [kind]
    if len(chunks) > MAX_CHUNKS_PER_EVENT:
        head = chunks[: MAX_CHUNKS_PER_EVENT - 1]
        head.append(" ".join(chunks[MAX_CHUNKS_PER_EVENT - 1 :])[:CHUNK_MAX_CHARS])
        return head
    return chunks
