"""Three silent losses on the write path.

Each one had the same shape: the consolidator dropped something a subject
had actually said, incremented a counter, and returned success. Nothing
raised, nothing logged at warning level, and no test failed -- so the loss
was only visible weeks later as an eval score that would not move.
"""

from app.models import FactStatus
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, facts_for, make_memory_event, run_worker


async def _active(subject: str, predicate: str):
    return [
        fact
        for fact in await facts_for(subject, predicate)
        if fact.status is FactStatus.active
    ]


# --------------------------------------------------------------------------
# P10a — a value coming back after being superseded
# --------------------------------------------------------------------------

async def test_returning_to_a_previous_value_supersedes_the_current_one(client):
    """A -> B -> A.

    `_find_duplicate` searches every non-deleted status, so it matched the
    SUPERSEDED A; `_reinforce_or_count_duplicate` saw a non-active fact,
    counted a duplicate and returned -- leaving B active. The subject said
    they went back to A and the ledger kept serving B.

    This is the knowledge-update failure mode LongMemEval has a whole
    category for.
    """
    for value, occurred, action in (
        ({"company": "Acme"}, "2026-01-05T10:00:00Z", "create"),
        # An explicit supersede, as the extractor emits for an update: this
        # is what leaves Acme `superseded` and Globex active, i.e. the
        # state the third message actually lands in.
        ({"company": "Globex"}, "2026-02-05T10:00:00Z", "supersede"),
        ({"company": "Acme"}, "2026-03-05T10:00:00Z", "create"),
    ):
        await capture(
            client,
            [
                make_memory_event(
                    [
                        mock_fact(
                            "employer", value, subject_id="usr_revert", action=action
                        )
                    ],
                    subject_id="usr_revert",
                    occurred_at=occurred,
                )
            ],
        )
        await run_worker()

    active = await _active("usr_revert", "employer")
    assert len(active) == 1
    # Verified against the unfixed code: it returned Globex here.
    assert active[0].value == {"company": "Acme"}


async def test_re_asserting_a_superseded_value_with_nothing_active_stays_a_duplicate(
    client,
):
    """The other half of the rule, so the fix stays narrow.

    Without something active to replace, a re-asserted historical value is
    not an update to anything -- it is the plain duplicate it always was.
    """
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"company": "Acme"}, subject_id="usr_dup")],
                subject_id="usr_dup",
            )
        ],
    )
    await run_worker()
    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"company": "Acme"}, subject_id="usr_dup")],
                subject_id="usr_dup",
                occurred_at="2026-08-02T10:00:00Z",
            )
        ],
    )
    await run_worker()

    facts = await facts_for("usr_dup", "employer")
    assert len(facts) == 1
    assert facts[0].reinforcement_count >= 1


# --------------------------------------------------------------------------
# P10b — a key that stopped being true
# --------------------------------------------------------------------------

async def test_a_supersede_carries_forward_the_keys_it_does_not_restate(client):
    """The behaviour the carry-forward exists for, kept intact.

    Measured failure it fixed: `adoption_agency_research` went from
    {target, status} to a bare {status} on supersede, and the target was
    only recoverable from the superseded version.
    """
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "research",
                        {"target": "adoption agencies", "status": "researching"},
                        subject_id="usr_merge",
                    )
                ],
                subject_id="usr_merge",
            )
        ],
    )
    await run_worker()
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "research",
                        {"status": "completed"},
                        subject_id="usr_merge",
                        action="supersede",
                    )
                ],
                subject_id="usr_merge",
                occurred_at="2026-08-02T10:00:00Z",
            )
        ],
    )
    await run_worker()

    [active] = await _active("usr_merge", "research")
    assert active.value == {"target": "adoption agencies", "status": "completed"}


async def test_an_explicit_null_removes_a_key(client):
    """The direction the merge had no way to express.

    Once a key was in a value, no later update could take it out: the fact
    kept serving something that had stopped being true. A null sentinel
    adds the inverse without giving back the carry-forward.
    """
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "trip",
                        {"city": "Lisbon", "companion": "Marc"},
                        subject_id="usr_drop",
                    )
                ],
                subject_id="usr_drop",
            )
        ],
    )
    await run_worker()
    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "trip",
                        {"companion": None},
                        subject_id="usr_drop",
                        action="supersede",
                    )
                ],
                subject_id="usr_drop",
                occurred_at="2026-08-02T10:00:00Z",
            )
        ],
    )
    await run_worker()

    [active] = await _active("usr_drop", "trip")
    assert active.value == {"city": "Lisbon"}


# --------------------------------------------------------------------------
# P9 — the anti-echo gate
# --------------------------------------------------------------------------

async def _serve_context(client, subject: str, query: str = "employer") -> None:
    """One /v1/context call, so the fact lands in a trace the gate reads."""
    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": subject,
            "query": query,
            "budget_tokens": 2000,
        },
    )
    assert response.status_code == 200


async def test_an_update_to_a_served_fact_is_not_treated_as_an_echo(
    client, monkeypatch
):
    """The gate killed exactly what it must not.

    Its own module measured the two populations it has to separate
    (scripts/check_semantic_threshold.py): rephrased-same-value pairs at
    0.002-0.187 cosine, genuine value updates at 0.030-0.158. They overlap
    completely, and both sit under the 0.28 gate -- so every legitimate
    update to a fact served in the last 20 packets was destroyed here, with
    a counter as its only trace.

    The distance threshold is forced wide open rather than relying on
    semantic proximity: FakeProvider's sha256 embeddings never cluster, so
    a threshold-driven bug is undetectable with them -- which is why none
    of this was caught before. Opening the gate fully isolates the rule
    that changed: WHICH candidates it may reject.
    """
    from app import consolidator

    monkeypatch.setattr(consolidator, "ANTI_ECHO_MAX_DISTANCE", 2.0)

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"company": "Acme"}, subject_id="usr_echo")],
                subject_id="usr_echo",
            )
        ],
    )
    await run_worker()
    await _serve_context(client, "usr_echo")

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "employer",
                        {"company": "Globex"},
                        subject_id="usr_echo",
                        action="supersede",
                    )
                ],
                subject_id="usr_echo",
                occurred_at="2026-08-02T10:00:00Z",
            )
        ],
    )
    await run_worker()

    active = await _active("usr_echo", "employer")
    assert len(active) == 1
    assert active[0].value == {"company": "Globex"}, (
        "the update was swallowed by the anti-echo gate"
    )


async def test_a_reformulation_of_a_served_fact_is_still_rejected(client, monkeypatch):
    """The mechanism itself, unchanged.

    An agent repeating a served fact back produces a `create`, not a
    replacement -- and that is what must never re-enter the ledger.
    """
    from app import consolidator

    monkeypatch.setattr(consolidator, "ANTI_ECHO_MAX_DISTANCE", 2.0)

    await capture(
        client,
        [
            make_memory_event(
                [mock_fact("employer", {"company": "Acme"}, subject_id="usr_echo2")],
                subject_id="usr_echo2",
            )
        ],
    )
    await run_worker()
    await _serve_context(client, "usr_echo2")

    await capture(
        client,
        [
            make_memory_event(
                [
                    mock_fact(
                        "workplace",
                        {"company": "Acme Corporation"},
                        subject_id="usr_echo2",
                    )
                ],
                subject_id="usr_echo2",
                occurred_at="2026-08-02T10:00:00Z",
            )
        ],
    )
    await run_worker()

    assert await facts_for("usr_echo2", "workplace") == []
