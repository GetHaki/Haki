"""Three non-determinism bugs found on 22 aout, all with the same shape:
Postgres was left to break a tie itself (physical heap order, or a random
uuid4), so the SAME stored content could retrieve or index differently
between two calls that changed nothing on purpose. Two identical installs
of the same corpus could end up serving different packets forever, with no
error anywhere -- see the comments on `_FACT_TIEBREAK`/`_EPISODE_TIEBREAK`
in app.context and the ORDER BY in
app.consolidator._merge_facts_into_chunk_index.

These tests insert Fact/EpisodeChunk rows directly rather than going
through the consolidator's extraction pipeline: forcing a genuine tie on
the vector axis requires two rows to share the EXACT SAME embedding, and
FakeProvider derives a fact's embedding from its own rendered text -- two
facts with different content never tie by accident. Direct construction is
the only way to make the tie a certainty instead of a hope.
"""

from datetime import datetime, timedelta, timezone

from app.consolidator import _merge_facts_into_chunk_index
from app.context import RETRIEVAL_TOP_K, build_context
from app.db import async_session
from app.models import EpisodeChunk, Event, Fact, FactStatus
from app.providers.fake import FakeProvider, _embed_one

TIED_TURNS = RETRIEVAL_TOP_K + 16  # 80: more than one CTE LIMIT can keep

ORG = "org_acme"
PROJECT = "prj_support"
AS_OF = datetime(2026, 6, 1, tzinfo=timezone.utc)
TIE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _StableReranker:
    """A reranker that never reorders anything.

    Constant scores + Python's stable sort keep the pool in the order it
    already had -- isolating the tie-break fix under test from the
    unrelated cross-encoder re-ordering (mechanism F-R), which would
    otherwise be a second source of order change inside RERANK_TOP_K.
    """

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


async def _insert_tied_facts(subject_id: str, count: int, *, valid_from) -> None:
    """`count` facts that tie exactly on similarity (identical embedding)
    and full-text (no search_text, so no fts match) -- everything the
    hybrid score sees. `valid_from` may be one value (recency also tied,
    isolating the CONTENT tie-break) or a callable of the index (recency
    varies, isolating the RECENCY tie-break)."""
    shared_embedding = _embed_one("shared_marker")
    async with async_session() as session:
        for i in range(count):
            fact = Fact(
                org_id=ORG,
                project_id=PROJECT,
                subject_id=subject_id,
                predicate=f"marker_{i:03d}",
                value={"i": i},
                status=FactStatus.active,
                confidence=0.9,
                valid_from=valid_from(i) if callable(valid_from) else valid_from,
            )
            fact.embedding = shared_embedding
            session.add(fact)
        await session.commit()


async def _packet_predicates(subject_id: str) -> set[str]:
    async with async_session() as session:
        packet, _token_count, _trace_id = await build_context(
            session,
            project_id=PROJECT,
            subject_id=subject_id,
            query="shared_marker",
            budget_tokens=20000,
            embedder=FakeProvider(),
            reranker=_StableReranker(),
            as_of=AS_OF,
        )
        await session.commit()
    return {f["predicate"] for f in packet["facts"]}


async def test_the_same_corpus_retrieves_identically_in_two_installs():
    """The bug: a bare `Fact.id` tie-break made the surviving 64-out-of-80
    depend on the RANDOM uuid4 each install's insert happened to assign,
    not on the facts themselves. Two subjects seeded with the exact same
    80 (predicate, value) pairs, tied on every scored axis, stand in for
    two independent installs of the same corpus -- their random ids are
    guaranteed different (assigned at INSERT time), same as two real
    installs would be.

    `_FACT_TIEBREAK` breaks the tie on recency (also tied here) then on
    content (predicate, value) -- neither depends on the id -- so both
    subjects must keep the exact same 64 rows.
    """
    await _insert_tied_facts("usr_install_a", TIED_TURNS, valid_from=TIE_TIME)
    await _insert_tied_facts("usr_install_b", TIED_TURNS, valid_from=TIE_TIME)

    kept_a = await _packet_predicates("usr_install_a")
    kept_b = await _packet_predicates("usr_install_b")

    assert len(kept_a) == RETRIEVAL_TOP_K
    # Ascending on a tied group (no .desc() on Fact.predicate in
    # _FACT_TIEBREAK): the CTE LIMIT keeps the alphabetically-first 64.
    expected = {f"marker_{i:03d}" for i in range(RETRIEVAL_TOP_K)}
    assert kept_a == expected, "install A did not cut the tie by content"
    assert kept_b == expected, "install B did not cut the tie by content"
    assert kept_a == kept_b, (
        "two installs of the identical corpus retrieved a different subset "
        "of a tied group -- the uuid4 tie-break regressed"
    )


async def test_a_tied_group_is_cut_by_recency_not_by_identifier():
    """Same tie on similarity/full-text as above, but recency now genuinely
    differs across the group (`_FACT_TIEBREAK`'s FIRST key). The LIMIT must
    keep the most RECENT 64, not the 64 that happen to sort first by
    predicate or by id -- proving recency is applied before content, not
    only as a fallback once content is also tied.
    """
    await _insert_tied_facts(
        "usr_recency_cut",
        TIED_TURNS,
        valid_from=lambda i: TIE_TIME + timedelta(hours=i),
    )

    kept = await _packet_predicates("usr_recency_cut")

    assert len(kept) == RETRIEVAL_TOP_K
    # Index i's recency is TIE_TIME + i hours: the most recent 64 are the
    # LAST 64 indices (16..79), not the first 64 (0..63).
    expected = {
        f"marker_{i:03d}" for i in range(TIED_TURNS - RETRIEVAL_TOP_K, TIED_TURNS)
    }
    assert kept == expected, (
        "the tied group was not cut by recency -- oldest survivors present, "
        "or newest missing"
    )


async def _chunk_index_text(event_subject: str, fact_order: list[int]) -> str:
    """One event, one chunk, five facts tied on `recorded_from` -- insert
    them in `fact_order` and return the resulting index_text."""
    event = Event(
        org_id=ORG,
        project_id=PROJECT,
        subject_type="user",
        subject_id=event_subject,
        kind="chat_session",
        occurred_at=TIE_TIME,
        payload={"messages": [{"role": "user", "content": "shared chunk text"}]},
        hash=f"sha256:{event_subject}",
        idempotency_key=f"idem-{event_subject}",
    )
    async with async_session() as session:
        session.add(event)
        await session.flush()
        chunk = EpisodeChunk(
            event_id=event.id,
            ordinal=0,
            project_id=PROJECT,
            subject_id=event_subject,
            occurred_at=event.occurred_at,
            origin_trust="trusted",
            text="shared chunk text",
            index_text="shared chunk text",
            embedding=None,
        )
        session.add(chunk)
        await session.flush()

        for i in fact_order:
            fact = Fact(
                org_id=ORG,
                project_id=PROJECT,
                subject_id=event_subject,
                predicate=f"detail_{i}",
                value={"i": i},
                status=FactStatus.active,
                confidence=0.9,
                recorded_from=TIE_TIME,
                source_event_ids=[event.id],
                source_chunk_id=chunk.id,
            )
            session.add(fact)
        await session.commit()

        await _merge_facts_into_chunk_index(session, event, FakeProvider())
        await session.commit()
        await session.refresh(chunk)
        return chunk.index_text


async def test_a_chunk_index_is_the_same_string_whatever_the_heap_says():
    """Five facts tied on `recorded_from`, inserted in two different
    orders for two otherwise-identical chunks. Without the explicit
    ORDER BY (recorded_from, predicate, value, id), which physical order
    Postgres happens to return them in decides the concatenation --
    unstable across a replay, hence a different embedding for the same
    content (measured: 86 of 231 packets differed between two runs with
    nothing changed). With it, insertion order cannot reach the result.
    """
    forward = await _chunk_index_text("usr_heap_fwd", [0, 1, 2, 3, 4])
    reverse = await _chunk_index_text("usr_heap_rev", [4, 3, 2, 1, 0])

    assert forward == reverse, "index_text depended on insertion order"
    for i in range(5):
        assert f"detail_{i}" in forward
