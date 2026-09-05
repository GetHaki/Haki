"""When a fact is ABOUT, as a typed column instead of free-form JSON.

Three different instants get confused constantly in a memory system, and
Haki stores three columns for them:

    recorded_from   when Haki learned it        (always known)
    valid_from      when it became true         (the message's timestamp)
    observed_at     when the fact HAPPENED      <- this module

"I got pre-approved for my mortgage back in August", said on 30 November:
recorded_from and valid_from are both 30 November. The answer to "when did
you get pre-approved?" is August, and until 21 aout that August lived
either as a free-form key inside `value` JSON or inside `temporal_range`,
in two different shapes, typed as nothing, indexed by nothing.

That matters more than it looks. Temporal reasoning is the category every
published memory system is worst at -- Mem0 55.5 % against 67.1 % on
single-hop, OpenAI's memory 21.7 % (arXiv 2504.19413, Table 1) -- and a
date that is not a date cannot be compared, ordered, filtered or rendered
consistently to the reader.

Extraction asks; extraction does not deliver
--------------------------------------------
The extraction prompt tells the extractor what to do, and it is right: an
ABSOLUTE date goes into `value` ({"date": "2023-06-02"}), a RELATIVE
expression is resolved into `temporal_range` anchored on the event
({"start": "2023-08-01", "end": "2023-08-31"}). Until 23 aout this module
trusted that, on the reasoning that the prompt already said so.

Measured on a real provider (gpt-4o-mini, 10 LoCoMo conversations, 220
active facts): **4 facts, 1.8 %, come back with a resolved
`temporal_range`**. "next month", "last week" are stored verbatim inside
`value`, as strings, exactly as the subject said them. The plumbing built
on 21 aout -- the typed column, the partial index, the dual-date rendering
-- sat on top of a field that is empty 98 % of the time.

So the resolution happens HERE, at consolidation, where `event.occurred_at`
is known and "next month" is arithmetic. Not in the prompt: a longer prompt
is paid on every single event, and a probabilistic model is the wrong tool
for a calculation a function does exactly. Nothing is removed from `value`
-- the reader still sees the JSON it always did.

Exact or absent
----------------
`temporal_range.start` wins when present: the extractor resolved it
explicitly and said so. Otherwise a date is taken from `value` only when
there is exactly ONE parseable one. Several dates in one value is
ambiguous, and a confidently wrong date is worse than no date at all --
it would be rendered to the reader as fact. Same rule as `source_chunk_id`
in the consolidator, for the same reason.

Key-agnostic on purpose: the extractor is free to call the key "date",
"when", "start_date", "purchased_on" or anything else, and matching on a
list of key names would silently miss the ones nobody thought of. What is
matched is the VALUE looking like an ISO date.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

# ISO 8601 date, optionally with a time. Anchored at both ends so that
# "chapter 2023-01-01 of the manual" or an id like "20230102" never
# accidentally reads as a date -- the extractor emits clean ISO dates
# (the prompt asks for them), and anything else is left alone.
_ISO_DATE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)


def parse_iso_instant(raw: Any) -> datetime | None:
    """A UTC datetime from an ISO date or datetime string, or None.

    A bare date becomes midnight UTC: a fact dated "2023-06-02" is about
    that day, and picking its start is the only choice that keeps
    comparisons between a date and a datetime meaningful.
    """
    if not isinstance(raw, str) or not _ISO_DATE.match(raw.strip()):
        return None
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text[:10]), time.min)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dates_in(value: Any) -> list[datetime]:
    """Every ISO instant reachable in a value, depth-first."""
    found: list[datetime] = []
    if isinstance(value, dict):
        for item in value.values():
            found += _dates_in(item)
    elif isinstance(value, list):
        for item in value:
            found += _dates_in(item)
    else:
        parsed = parse_iso_instant(value)
        if parsed is not None:
            found.append(parsed)
    return found


# Relative expressions with an EXACT calendar meaning, and only those. The
# module's rule everywhere else is "exact or absent", and it applies with
# more force here: a date resolved from a vague phrase is rendered to the
# reader as a fact. "recently", "a few weeks ago", "a while back", "soon"
# are deliberately NOT in this table and never will be -- they have no
# exact referent, and guessing one is how a memory system starts lying
# confidently.
#
# Each entry maps to (unit, offset). The unit decides the RANGE: a month
# expression is about the whole month, a day expression about one day.
_RELATIVE_UNITS = ("day", "week", "month", "year")
_RELATIVE_PHRASES: dict[str, tuple[str, int]] = {
    "yesterday": ("day", -1),
    "today": ("day", 0),
    "tonight": ("day", 0),
    "tomorrow": ("day", 1),
    "last week": ("week", -1),
    "this week": ("week", 0),
    "next week": ("week", 1),
    "last month": ("month", -1),
    "this month": ("month", 0),
    "next month": ("month", 1),
    "last year": ("year", -1),
    "this year": ("year", 0),
    "next year": ("year", 1),
}
_PHRASE_RE = re.compile(
    r"\b(" + "|".join(sorted(_RELATIVE_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# "3 weeks ago", "in 2 months". A written-out number is not accepted: "a
# few" and "a couple of" are exactly the vague cases above, and "two" would
# open the door to them.
_COUNTED_RE = re.compile(
    r"\b(?:(\d{1,3})\s+(day|week|month|year)s?\s+ago"
    r"|in\s+(\d{1,3})\s+(day|week|month|year)s?)\b",
    re.IGNORECASE,
)


def _add_months(anchor: date, months: int) -> date:
    total = anchor.year * 12 + (anchor.month - 1) + months
    year, month = divmod(total, 12)
    return date(year, month + 1, min(anchor.day, calendar.monthrange(year, month + 1)[1]))


def _range_for(anchor: date, unit: str, offset: int) -> tuple[date, date]:
    """The span an expression covers, not just its first instant.

    "next month" is about a month, not about its first day. Keeping the end
    is what lets a reader answer "was it in August?" instead of only "was it
    on 1 August?".
    """
    if unit == "day":
        moved = anchor + timedelta(days=offset)
        return moved, moved
    if unit == "week":
        moved = anchor + timedelta(weeks=offset)
        start = moved - timedelta(days=moved.weekday())
        return start, start + timedelta(days=6)
    if unit == "month":
        moved = _add_months(anchor, offset)
        return (
            moved.replace(day=1),
            moved.replace(day=calendar.monthrange(moved.year, moved.month)[1]),
        )
    moved = date(anchor.year + offset, 1, 1)
    return moved, date(moved.year, 12, 31)


def _strings_in(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [t for item in value.values() for t in _strings_in(item)]
    if isinstance(value, list):
        return [t for item in value for t in _strings_in(item)]
    return [value] if isinstance(value, str) else []


def resolve_relative_range(value: Any, anchor: datetime | None) -> dict[str, str] | None:
    """`{"start": iso, "end": iso}` for a relative expression in `value`.

    None whenever the answer would be a guess: no anchor, no expression,
    an expression outside the exact table above, or -- the case that
    matters most -- MORE THAN ONE distinct expression. Two relative
    expressions in one value ("we met last week and I'm going back next
    month") name two different instants and nothing here can tell which one
    the fact is about. Same rule as `source_chunk_id` and as the single-ISO
    rule below, for the same reason: a confidently wrong date is worse than
    no date at all, because it reaches the reader as a fact.
    """
    if anchor is None:
        return None
    found: set[tuple[str, int]] = set()
    for text in _strings_in(value):
        for phrase in _PHRASE_RE.findall(text):
            found.add(_RELATIVE_PHRASES[phrase.lower()])
        for past_n, past_unit, future_n, future_unit in _COUNTED_RE.findall(text):
            if past_n:
                found.add((past_unit.lower(), -int(past_n)))
            else:
                found.add((future_unit.lower(), int(future_n)))
        if len(found) > 1:
            return None
    if len(found) != 1:
        return None
    unit, offset = found.pop()
    start, end = _range_for(anchor.astimezone(timezone.utc).date(), unit, offset)
    return {"start": start.isoformat(), "end": end.isoformat()}


def observed_at_of(value: Any, temporal_range: dict | None) -> datetime | None:
    """When the fact is about, or None when that cannot be known exactly.

    None is a real answer, not a failure: most facts ("I have a dog") are
    about no particular instant, and inventing one for them would make the
    column meaningless for the ones that do.

    Priority (B10): a single explicit ISO date inside `value` wins over the
    range's start. The range is a resolved relative phrase ("last week") --
    the extractor's understanding of a vague expression -- while an ISO
    date in the value is the precise instant itself (same hierarchy as
    _apply_candidate's "an extractor-resolved range ... a plain ISO date
    already inside value is more precise than any phrase"). Before this,
    the range always won, so value={date: 2023-06-20} with a "last week"
    range resolved to observed_at=range start instead of June 20th.
    """
    dates = _dates_in(value)
    if len(dates) == 1:
        return dates[0]
    if isinstance(temporal_range, dict):
        return parse_iso_instant(temporal_range.get("start"))
    return None
