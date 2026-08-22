"""`as_of` is a point of view: nothing dated after it may be served.

The bug this pins (found 20 Aug)
------------------------------------
`build_context` filtered only the END of a fact's validity interval
(`valid_to IS NULL OR valid_to > now`) and never its START. A fact whose
validity begins AFTER the caller's point of view was therefore served as
if it were already known.

Two consequences, both real on every `as_of` run — which means every eval
run, since the harness always passes one:

1. **Temporal leak.** A question dated T could be answered from a fact
   dated T+n. The temporal-reasoning categories of LoCoMo and
   LongMemEval exist precisely to catch that.
2. **Ranking corruption.** The recency term is
   `exp(-(now - valid_from) / tau)`. When `valid_from > now` the exponent
   turns positive: `exp(+1) = 2.72` at 30 days ahead, `exp(+3) = 20` at
   90. Weighted 0.15, that single term outweighs similarity (0.6) and
   full-text (0.25) combined, so future-dated facts sat at the top of the
   ranking and ate the token budget.

The same reasoning applies to episodes, whose clock is `occurred_at` —
including the "after" temporal-neighbour bolt-on, the one place in
`build_context` that deliberately looks forward in time.
"""

from datetime import datetime, timedelta, timezone

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

QUERY = {"project_id": "prj_support", "subject_id": "usr_42", "budget_tokens": 2000}


async def _context(client, query: str, as_of: datetime):
    response = await client.post(
        "/v1/context", json={**QUERY, "query": query, "as_of": as_of.isoformat()}
    )
    assert response.status_code == 200
    return response.json()


async def test_a_fact_dated_after_as_of_is_never_served(client):
    past = datetime(2026, 3, 1, tzinfo=timezone.utc)
    future = datetime(2026, 9, 1, tzinfo=timezone.utc)
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"company": "Acme"})],
                occurred_at=past.isoformat(),
            ),
            make_memory_event(
                [mock_fact("future_employer", {"company": "Globex"})],
                occurred_at=future.isoformat(),
            ),
        ],
    )
    await run_worker()

    body = await _context(client, "employer company", as_of=past + timedelta(days=1))
    predicates = {fact["predicate"] for fact in body["packet"]["facts"]}
    assert "employer" in predicates
    assert "future_employer" not in predicates, (
        "a fact whose validity starts after the caller's point of view was served"
    )


async def test_a_future_fact_does_not_outrank_a_relevant_past_one(client):
    """The ranking half of the same bug, isolated.

    Before the fix the future-dated fact scored `exp(+x)` on recency and
    won the pool outright, even against a fact the query actually matches.
    Here the budget is deliberately generous: the assertion is about
    ORDER, not about what fits.
    """
    past = datetime(2026, 3, 1, tzinfo=timezone.utc)
    far_future = datetime(2027, 3, 1, tzinfo=timezone.utc)
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("kayak_owner", {"item": "kayak"})],
                occurred_at=past.isoformat(),
            ),
            make_memory_event(
                [mock_fact("unrelated_future", {"item": "telescope"})],
                occurred_at=far_future.isoformat(),
            ),
        ],
    )
    await run_worker()

    body = await _context(client, "kayak", as_of=past + timedelta(days=1))
    predicates = [fact["predicate"] for fact in body["packet"]["facts"]]
    assert predicates, "expected at least the matching fact"
    assert predicates[0] == "kayak_owner"
    assert "unrelated_future" not in predicates


async def test_an_episode_dated_after_as_of_is_never_served(client):
    """Episodes are filtered on `occurred_at` for the same reason.

    An episode is a verbatim payload excerpt replayed into the agent's
    context; serving one from the future is the same leak as a fact, with
    the original wording attached.
    """
    past = datetime(2026, 3, 1, tzinfo=timezone.utc)
    future = datetime(2026, 9, 1, tzinfo=timezone.utc)
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("kayak_owner", {"item": "kayak"})],
                occurred_at=past.isoformat(),
            ),
            make_memory_event(
                [mock_fact("later_note", {"item": "kayak"})],
                occurred_at=future.isoformat(),
            ),
        ],
    )
    await run_worker()

    body = await _context(client, "kayak", as_of=past + timedelta(days=1))
    for episode in body["packet"].get("episodes", []):
        assert episode["occurred_at"] <= (past + timedelta(days=1)).isoformat(), (
            f"episode from the future served: {episode['occurred_at']}"
        )
