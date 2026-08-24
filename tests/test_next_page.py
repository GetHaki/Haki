"""The next page of the same ranked list -- not a second hop.

`exclude_ids` exists because of one measurement. On the questions whose
first packet holds only PART of their evidence, simulating a second call:

    same query, seen items excluded              44.8 % find the missing turn
    query + what the first packet contained      41.4 %
    that content alone (the most generous form)  27.6 %

Rewriting the query with what the agent just read is measurably WORSE than
asking again unchanged. So this is not iterative reasoning and the API must
not invite it: it serves further down the same ranked list, which is the
same thing as a bigger budget -- paid only by the callers who need it.

That is also why these tests check the boring properties (nothing served
twice, the same query still works, a bad id cannot break a read) rather
than any claim about multi-hop.
"""

import uuid

from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

QUERY = "Zolgorvex kayak"


async def _seed(client) -> None:
    # Enough facts for TWO full pages at budget=400 -- the first call packs
    # ~34 of these at ~11.7 tokens each (measured), so anything fewer than
    # ~70 starves the second page by construction, not by a real bug.
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(f"owns_kayak_{i}", {"model": i, "brand": "Zolgorvex"})
                    for i in range(90)
                ]
            )
        ],
    )
    await run_worker()


async def _context(client, *, exclude=None, budget=400):
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": QUERY,
            "budget_tokens": budget,
            **({"exclude_ids": exclude} if exclude is not None else {}),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ids(body) -> list[str]:
    packet = body["packet"]
    return [f["id"] for f in packet["facts"]] + [
        e["episode_id"] for e in packet["episodes"] if e.get("episode_id")
    ]


async def test_the_second_page_never_repeats_the_first(client):
    await _seed(client)
    first = await _context(client)
    assert _ids(first), "nothing served -- the test would pass vacuously"

    second = await _context(client, exclude=_ids(first))
    assert _ids(second), "the second page is empty -- there was more to serve"
    assert not set(_ids(first)) & set(_ids(second))


async def test_the_second_page_fills_the_budget_like_the_first(client):
    """The exclusion happens at candidate generation, not after packing.

    Excluding after ranking would still spend the top-K slots on rows the
    caller is about to throw away, and the second page would come back
    thin. With material left in the scope, it comes back full.
    """
    await _seed(client)
    first = await _context(client)
    second = await _context(client, exclude=_ids(first))
    assert len(_ids(second)) >= len(_ids(first)) - 1
    assert second["token_count"] >= first["token_count"] * 0.8


async def test_the_ids_the_packet_gives_are_the_ids_the_caller_sends_back(client):
    """`seen(packet)` must be a correct implementation of this contract."""
    import sys

    sys.path.insert(0, "sdk/python/src")
    from haki.runtime import seen

    await _seed(client)
    first = await _context(client)
    assert sorted(seen(first)) == sorted(_ids(first))
    second = await _context(client, exclude=seen(first))
    assert not set(seen(first)) & set(seen(second))


async def test_a_malformed_id_does_not_fail_the_read(client):
    """A caller's typo is not a reason to refuse to remember.

    The packet is still correct; it merely holds an item they meant to
    skip. Failing the call would turn a cosmetic mistake into an agent with
    no memory at all.
    """
    await _seed(client)
    body = await _context(client, exclude=["not-a-uuid", "", str(uuid.uuid4())])
    assert body["packet"]["facts"] or body["packet"]["episodes"]


async def test_excluding_nothing_changes_nothing(client):
    await _seed(client)
    plain = await _context(client)
    empty = await _context(client, exclude=[])
    assert _ids(plain) == _ids(empty)


async def test_an_event_id_excludes_nothing(client):
    """Documented on purpose: an event names a whole session.

    Dropping every turn of a session because one of them was served is not
    what "I already have this" means, and a caller who passes the wrong
    field should get a slightly redundant packet, not a silently empty one.
    """
    await _seed(client)
    first = await _context(client)
    event_ids = [e["event_id"] for e in first["packet"]["episodes"]]
    if not event_ids:
        return
    again = await _context(client, exclude=event_ids)
    assert set(_ids(again)) == set(_ids(first))
