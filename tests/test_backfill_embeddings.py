"""scripts/backfill_embeddings.py: re-embed what migration 0029 NULLed.

The migration widens the `embedding` columns to vector(1024) and cannot
compute new vectors itself -- that needs the embedder loaded at runtime.
These tests exercise the real `backfill`/`status` functions against real
rows (no mock of the DB or the embedder), covering the two ways this
would go wrong silently: an interrupted run that never finishes (resumed
by re-running, not by any checkpoint state), and a row with no source
text that would otherwise make `WHERE embedding IS NULL` pick the same
row forever.
"""

from datetime import UTC, datetime

from sqlalchemy import text

from app.db import async_session
from app.models import EpisodeChunk, Event, Fact, FactStatus
from scripts.backfill_embeddings import _ZERO_VECTOR, backfill, status

ORG = "org_acme"
PROJECT = "prj_support"
SUBJECT = "usr_backfill"


async def _restore_embedding_space() -> None:
    async with async_session() as session:
        await session.execute(
            text("UPDATE embedding_space SET backfilled_at = NULL WHERE id = 1")
        )
        await session.commit()


async def _seed_with_text() -> tuple:
    """One Fact, one Event, one EpisodeChunk -- embedding=NULL, real source
    text -- exactly what a fresh migration 0029 leaves behind."""
    async with async_session() as session:
        fact = Fact(
            org_id=ORG,
            project_id=PROJECT,
            subject_id=SUBJECT,
            predicate="favorite_color",
            value={"color": "blue"},
            status=FactStatus.active,
            confidence=0.9,
            search_text="favorite_color: blue",
        )
        event = Event(
            org_id=ORG,
            project_id=PROJECT,
            subject_type="user",
            subject_id=SUBJECT,
            kind="chat_session",
            occurred_at=datetime.now(UTC),
            payload={"messages": [{"role": "user", "content": "hello"}]},
            hash="sha256:backfill-test",
            idempotency_key="idem-backfill-test",
            index_text="hello",
        )
        session.add_all([fact, event])
        await session.flush()
        chunk = EpisodeChunk(
            event_id=event.id,
            ordinal=0,
            project_id=PROJECT,
            subject_id=SUBJECT,
            occurred_at=event.occurred_at,
            origin_trust="trusted",
            text="hello",
            index_text="hello",
        )
        session.add(chunk)
        await session.commit()
        return fact.id, event.id, chunk.id


async def test_it_embeds_every_row_across_all_three_tables():
    fact_id, event_id, chunk_id = await _seed_with_text()

    await backfill(batch=200)

    async with async_session() as session:
        fact = await session.get(Fact, fact_id)
        event = await session.get(Event, event_id)
        chunk = await session.get(EpisodeChunk, chunk_id)
    assert fact.embedding is not None
    assert event.embedding is not None
    assert chunk.embedding is not None
    await _restore_embedding_space()


async def test_it_is_resumable_and_idempotent():
    """A second pass over an already-embedded corpus changes nothing and
    does not error -- `WHERE embedding IS NULL` is what makes an
    interrupted run safe to just run again, with no checkpoint file."""
    fact_id, _event_id, _chunk_id = await _seed_with_text()
    await backfill(batch=200)
    async with async_session() as session:
        before = (await session.get(Fact, fact_id)).embedding

    await backfill(batch=200)  # must not raise, must not touch this row again

    async with async_session() as session:
        after = (await session.get(Fact, fact_id)).embedding
    assert before == after
    await _restore_embedding_space()


async def test_status_reports_pending_without_writing(capsys):
    fact_id, _event_id, _chunk_id = await _seed_with_text()

    await status()

    async with async_session() as session:
        fact = await session.get(Fact, fact_id)
    assert fact.embedding is None  # --status must never write
    out = capsys.readouterr().out
    assert "facts" in out and "pending=1" in out


async def test_a_row_with_empty_source_text_gets_a_zero_vector_not_an_infinite_loop():
    async with async_session() as session:
        fact = Fact(
            org_id=ORG,
            project_id=PROJECT,
            subject_id=SUBJECT,
            predicate="no_text",
            value={},
            status=FactStatus.active,
            confidence=0.9,
            search_text=None,
        )
        session.add(fact)
        await session.commit()
        fact_id = fact.id

    await backfill(batch=200)  # a NULL search_text must not loop forever

    async with async_session() as session:
        fact = await session.get(Fact, fact_id)
    assert fact.embedding == _ZERO_VECTOR
    await _restore_embedding_space()
