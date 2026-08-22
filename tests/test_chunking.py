"""Cutting an event into retrievable units.

The unit tests pin the contract of the cutter itself; the end-to-end test
pins the capability it exists for -- serving one relevant turn of a long
session under a budget that could never have held the session.
"""

import uuid

from app.context.chunking import (
    CHUNK_MAX_CHARS,
    MAX_CHUNKS_PER_EVENT,
    chunk_payload,
)


def test_a_message_list_is_cut_on_turn_boundaries():
    payload = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    assert chunk_payload("chat_session", payload) == ["user: hi", "assistant: hello"]


def test_the_speaker_is_kept_in_the_chunk():
    """Who said it is the single most useful token in a two-party log.

    It is what tells "Caroline said" from "Melanie said" -- the failure
    mode the entity mechanism in app.context exists for -- and dropping it
    would make the two speakers' turns lexically indistinguishable.
    """
    [chunk] = chunk_payload("chat_session", {"messages": [{"speaker": "Caroline", "text": "I went"}]})
    assert chunk.startswith("Caroline: ")


def test_an_event_with_no_message_structure_stays_one_chunk():
    """A short opaque payload must not be split for the sake of splitting."""
    chunks = chunk_payload("conversation.message", {"role": "user", "content": "short"})
    assert len(chunks) == 1
    assert "conversation.message" in chunks[0]


def test_a_long_opaque_payload_is_split_without_losing_text():
    """No silent truncation. Ever.

    This is the regression that matters: the previous unit truncated at
    4 000 characters, which destroyed 7.1 % of the eval corpus outright
    (69 of 272 sessions). Whatever the cutter does, the concatenation of
    its output must still contain everything it was given.
    """
    body = ". ".join(f"sentence number {i} with some filler words" for i in range(400))
    chunks = chunk_payload("note", {"body": body})
    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks)
    for i in (0, 200, 399):
        assert f"sentence number {i}" in " ".join(chunks)


def test_an_empty_payload_still_produces_one_chunk():
    """An event with no episodes is an event that can never be retrieved."""
    assert chunk_payload("ping", {}) != []
    assert chunk_payload("ping", None) != []


def test_the_chunk_count_is_bounded_and_the_tail_is_merged_not_dropped():
    payload = {"messages": [{"role": "user", "content": f"turn {i}"} for i in range(MAX_CHUNKS_PER_EVENT + 50)]}
    chunks = chunk_payload("chat_session", payload)
    assert len(chunks) == MAX_CHUNKS_PER_EVENT
    assert f"turn {MAX_CHUNKS_PER_EVENT + 49}" in chunks[-1]


def test_chunking_is_deterministic():
    """The consolidator skips an event that already has chunks, which is
    only sound because the same payload always yields the same chunks."""
    payload = {"messages": [{"role": "user", "content": f"turn {i}"} for i in range(20)]}
    assert chunk_payload("chat_session", payload) == chunk_payload("chat_session", payload)


async def test_one_relevant_turn_of_a_long_session_is_served_under_a_tight_budget(client):
    """The capability the whole change exists for.

    A twenty-turn session, one of which mentions a rare term. The budget is
    60 tokens -- the session as a single episode cost ~810 and could not
    have been served at all, let alone leave room for anything else.
    """
    session_payload = {
        "messages": [
            {"role": "user", "content": f"We talked about ordinary thing number {i}."}
            for i in range(20)
        ]
    }
    session_payload["messages"][13] = {
        "role": "user",
        "content": "I finally bought the Zolgorvex kayak this weekend.",
    }
    response = await client.post(
        "/v1/capture",
        json={
            "idempotency_key": f"batch-{uuid.uuid4()}",
            "events": [
                {
                    "org_id": "org_acme",
                    "project_id": "prj_support",
                    "subject_type": "user",
                    "subject_id": "usr_chunked",
                    "kind": "chat_session",
                    "occurred_at": "2026-05-01T10:00:00Z",
                    "payload": session_payload,
                }
            ],
        },
    )
    assert response.status_code == 202

    from tests.test_consolidator import run_worker

    await run_worker()

    context = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_chunked",
            "query": "Zolgorvex",
            "budget_tokens": 60,
        },
    )
    assert context.status_code == 200
    body = context.json()
    excerpts = " ".join(episode["excerpt"] for episode in body["packet"]["episodes"])
    assert "Zolgorvex" in excerpts, (
        f"the matching turn was not served under a 60-token budget: {excerpts!r}"
    )
    assert body["token_count"] <= 60
