"""What a packet item costs in the prompt it will end up in.

Until now the budget was charged on a STRIPPED string -- `predicate value`
for a fact, `date kind excerpt` for an episode -- while the caller's prompt
carried the fully rendered line. Measured on eval.retrieval_bench (LoCoMo
1-2, n=231, o200k tokenizer):

    charged  (estimate_tokens on the stripped string)   43.3 tokens/item
    real     (the same stripped string)                 43.9 tokens/item
    real     (the line the reader actually receives)    93.4 tokens/item

So `budget_tokens=2000` put a median of 4565 tokens into the caller's
prompt: 2.28x what they asked for, on every call. Note the first two rows:
the len//4 heuristic itself is accurate to 1 % against a real tokenizer, so
the fix is NOT to tokenize properly -- it is to charge for the right
string. estimate_tokens stays as it is.

`render_line` mirrors the SDK's build_prompt_context exactly, and
tests/test_packet_cost.py asserts STRING equality between the two on
packets carrying every marker. Two renderers is a real risk -- a second,
poorer copy of this one is what P14 found inside the MCP server -- so the
contract is pinned by equality, not by "close enough".
"""

from typing import Any

# 3.5 chars per token, measured on the lines this module renders, over the
# 231 packets of eval.retrieval_bench: 3.60 under o200k_base (GPT-4o and
# later), 3.54 under cl100k_base. The old //4 was calibrated against the
# STRIPPED string and under-counts the rendered one by 11-13 % -- brackets,
# ISO dates and punctuation tokenize far worse than prose.
#
# Not tiktoken. A budget served to Claude, Llama or Mistral cannot be exact
# under an OpenAI tokenizer, so the goal is to be UNBIASED, not exact, and
# a divisor costs neither a 2 MB dependency nor a per-model coupling. 3.5
# rather than 3.6 so the estimate errs 1-3 % HIGH: a budget the caller sets
# should be a ceiling they stay under, not one they discover they crossed.
_CHARS_PER_TOKEN = 3.5

# The fixed instruction blocks are plain English prose and tokenize much
# better than a data-dense line: 4.63 chars/token, IDENTICAL under o200k
# and cl100k (they are one constant text, so this is a measurement of that
# text, not an average over a corpus). Charging them at 3.5 would
# over-count them by 32 % -- one wrong divisor replacing another.
_CHARS_PER_TOKEN_PROSE = 4.6


def estimate_tokens(text: str) -> int:
    """Cost of a rendered ITEM line -- see _CHARS_PER_TOKEN."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_prose_tokens(text: str) -> int:
    """Cost of one of the fixed instruction blocks below."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN_PROSE))


def _fact_valid_from(fact: dict[str, Any]) -> str:
    valid_from = fact.get("valid_from_short") or fact.get("valid_from") or "unknown date"
    relative = fact.get("valid_from_relative")
    if relative:
        valid_from = f"{valid_from} — {relative}"
    temporal_range = fact.get("temporal_range")
    if temporal_range:
        valid_from += (
            f"; described event dated {temporal_range.get('start')} to "
            f"{temporal_range.get('end')}"
        )
    return valid_from


def _fact_marker(fact: dict[str, Any]) -> str:
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
    if fact.get("auto_reclassified"):
        marker += (
            " [AUTO-RECLASSIFIED: the system automatically decided this is "
            "one occurrence among several for this subject, based on 3+ "
            "differing values arriving for what looked like a single "
            "attribute — if these look like updates to ONE attribute over "
            "time (e.g. successive employers) rather than genuinely "
            "distinct occurrences (e.g. separate volunteering events), "
            "flag this to the subject instead of treating all values as "
            "equally current]"
        )
    return marker


def _episode_marker(episode: dict[str, Any]) -> str:
    return (
        " [surrounding context — not independently matched to the "
        "query, included for the conversational moment around a "
        "result above]"
        if episode.get("context_neighbor")
        else ""
    )


def render_line(kind: str, item: dict[str, Any]) -> str:
    """The exact line build_prompt_context will emit for this item.

    Two things left this string on purpose (22 aout), both measured on the
    real blocks of eval.retrieval_bench:

    - the uuid. A uuid4 is 35 o200k tokens; `E7` is 2. At ~46 identifiers
      per packet that was 1 068 tokens, 23 % of everything sent, spent on
      strings the reader is asked to cite and cannot usefully carry. The
      full ids stay in the packet JSON, where the caller resolves a ref
      exactly -- and a fact now cites ITSELF (`F3`) rather than its source
      event, which is a better citation anyway: it names what was used, not
      a provenance chain the reader never saw.
    - the seconds and the UTC offset of every timestamp, 6 % more.

    Together: 4 583 -> 3 316 tokens per packet, the SAME evidence, -27.6 %.
    """
    ref = item.get("ref") or "?"
    if kind == "fact":
        return (
            f"- [{ref}] {item.get('predicate')}: {item.get('value')} "
            f"(valid from {_fact_valid_from(item)}){_fact_marker(item)}"
        )
    occurred = item.get("occurred_at_short") or item.get("occurred_at") or "unknown date"
    relative = item.get("occurred_at_relative")
    if relative:
        occurred = f"{occurred} — {relative}"
    return (
        f"- [{ref}] [{occurred}] {item.get('kind')}: {item.get('excerpt')}"
        f"{_episode_marker(item)}"
    )


def short_timestamp(iso: str | None) -> str | None:
    """`2023-05-14T10:32:07+00:00` -> `2023-05-14 10:32`.

    Seconds and the UTC offset answer no question a memory packet is asked
    and cost 6 % of it. The date and the time of day can both matter, so
    both stay. Computed HERE, server-side, and carried in the packet as
    `*_short`, so there is one implementation of it and an older SDK simply
    falls back to the full ISO string it already knows.
    """
    if not iso:
        return None
    head, sep, _ = iso.partition("T")
    if not sep:
        return iso
    rest = _
    time_part = rest[:5]
    return f"{head} {time_part}" if len(time_part) == 5 and time_part[2] == ":" else head


# The instruction paragraphs of the block, canonical here because the
# budget has to charge them (22 aout). They were 289 tokens the caller
# never asked for and never saw counted -- 11 % of a 2 500-token block, on
# every call. Charged incrementally in build_context: the header when
# something is packed, the episodes paragraph when the first turn is
# packed, the contested chain-of-note only when a conflict is actually
# served (its own comment in the SDK already said "only paid when a
# conflict is being served" -- it just was not paid FOR).
#
# They live next to render_line for the same reason render_line lives here:
# one text, one cost, one place. An SDK that does not receive `preamble`
# falls back to its own copy, which tests/test_packet_cost.py pins to this
# one string for string.
HEADER = (
    "Verified long-term memory facts about this subject. You MUST apply them "
    "whenever they are relevant to the request: treat them as instructions "
    "about HOW to respond (language of your answer, format, constraints, "
    "decisions already made), not as background trivia. If a fact states a "
    "language preference, write your entire response in that language. "
    "Cite an item by the reference in square brackets at the start of "
    "its line (F3, E7) when you rely on it. Facts already reflect the "
    "CURRENT, resolved truth — an outdated value is removed the moment a "
    "newer one is confirmed, so you never need to compare dates between "
    "facts yourself; do not second-guess a fact's value. EXCEPTION: a fact "
    "marked CONTESTED below is an unresolved disagreement — for those, and "
    "only those, compare 'valid from' dates yourself and treat the most "
    "recent one as current. Relevance check first: these items were picked "
    "by similarity, so they can look topical while answering a different "
    "question. Before using any of them, check that at least one is even "
    "about what is asked — the same attribute, event, or decision. "
    "Competing candidates are fine (that is what the rest of these "
    "instructions resolve); inventing from nothing is not. If nothing "
    "below is even about it, say the memory does not contain it instead "
    "of promoting the closest-looking item to an answer."
)

# A minimal terse rule under-performs a spelled-out chain of steps for this
# exact task (Bug 3, 13 aout: gpt-4o-mini went 2/3 with a one-line rule,
# 3/3 with these same three steps as a worked chain-of-note).
CONTESTED_INSTRUCTIONS = (
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

EPISODES_HEADER = (
    "Dated events from the source history (episodic memory): raw excerpts "
    "kept for citation and narrative detail. They can mention values that "
    "were later updated — if anything here conflicts with a fact above, "
    "the FACT is the current, correct answer; never prefer an older "
    "mention from here over it. If two dated items disagree and no fact "
    "above covers it, use the one with the most recent date — UNLESS the "
    "question explicitly asks about a past/previous state, in which case "
    "use the one matching that earlier point in time instead."
)

# The delimiters and the newlines between blocks: small, constant, and part
# of what the caller pays.
WRAPPER = "<haki_memory>\n</haki_memory>"
