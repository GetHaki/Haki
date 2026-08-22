"""When a fact is ABOUT, as a typed column instead of free-form JSON.

Three different instants get confused constantly in a memory system, and
Haki stores three columns for them:

    recorded_from   when Haki learned it        (always known)
    valid_from      when it became true         (the message's timestamp)
    observed_at     when the fact HAPPENED      <- this module

"I got pre-approved for my mortgage back in August", said on 30 November:
recorded_from and valid_from are both 30 November. The answer to "when did
you get pre-approved?" is August, and until 21 Aug that August lived
either as a free-form key inside `value` JSON or inside `temporal_range`,
in two different shapes, typed as nothing, indexed by nothing.

That matters more than it looks. Temporal reasoning is the category every
published memory system is worst at -- Mem0 55.5 % against 67.1 % on
single-hop, OpenAI's memory 21.7 % (arXiv 2504.19413, Table 1) -- and a
date that is not a date cannot be compared, ordered, filtered or rendered
consistently to the reader.

Extraction, unchanged
----------------------
The extraction prompt already tells the extractor what to do, and it is
right: an ABSOLUTE date goes into `value` ({"date": "2023-06-02"}), a
RELATIVE expression is resolved into `temporal_range` anchored on the
event ({"start": "2023-08-01", "end": "2023-08-31"}). Nothing here asks
the extractor for anything new. This module reads what it already
produces and normalises it into one typed column, WITHOUT removing it from
`value` -- the reader sees the same JSON it always did.

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

import re
from datetime import date, datetime, time, timezone
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


def observed_at_of(value: Any, temporal_range: dict | None) -> datetime | None:
    """When the fact is about, or None when that cannot be known exactly.

    None is a real answer, not a failure: most facts ("I have a dog") are
    about no particular instant, and inventing one for them would make the
    column meaningless for the ones that do.
    """
    if isinstance(temporal_range, dict):
        start = parse_iso_instant(temporal_range.get("start"))
        if start is not None:
            return start
    dates = _dates_in(value)
    return dates[0] if len(dates) == 1 else None
