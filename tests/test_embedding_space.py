"""`app.db.verify_embedding_space`: the model-identity guard (migration 0028).

Same dimension does not mean the same embedding space. Two models can both
be 1024-dimensional and produce vectors that are not comparable -- cosine
similarity between them is noise, not a degraded signal, so a mismatch has
to REFUSE to start rather than warn. A pending backfill is the opposite
shape: partial and recoverable, so it only warns (tested separately from
test_startup_guards.py's `lifespan` guards, which cover provider selection
rather than this specific invariant).
"""

import logging

import pytest
from sqlalchemy import text

from app.config import settings
from app.db import async_session, verify_embedding_space
from app.providers import EMBEDDING_DIM

# Real, 384-dimensional -- the model this schema ran before migration 0029.
WRONG_DIM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Fictional: does not need to exist in app.providers.local.MODELS, because
# the model-identity check compares the STORED name against
# HAKI_EMBED_MODEL directly -- it does not need to know the model to catch
# that they differ.
SAME_DIM_OTHER_MODEL = "some-other-model-of-the-same-width"


async def _set_embedding_space(model: str, dim: int, backfilled: bool) -> None:
    async with async_session() as session:
        await session.execute(
            text(
                "UPDATE embedding_space SET model = :model, dim = :dim, "
                "backfilled_at = CASE WHEN :backfilled THEN now() ELSE NULL END "
                "WHERE id = 1"
            ),
            {"model": model, "dim": dim, "backfilled": backfilled},
        )
        await session.commit()


@pytest.fixture(autouse=True)
async def restore_embedding_space():
    """embedding_space is a singleton row, not truncated between tests
    (clean_tables leaves it alone -- it describes the deployment, not a
    project's data). Restore the state migration 0029 leaves it in so a
    test that mutates it cannot leak into the next one."""
    yield
    await _set_embedding_space(settings.embed_model, EMBEDDING_DIM, backfilled=False)


async def test_accepts_a_matching_model_with_a_complete_backfill(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "local")
    await _set_embedding_space(settings.embed_model, EMBEDDING_DIM, backfilled=True)
    await verify_embedding_space()  # must not raise


async def test_refuses_a_different_model_of_the_same_dimension(monkeypatch):
    """The hole 0028 exists for: two models, same width, incomparable
    vectors -- nothing else in the schema can tell them apart."""
    monkeypatch.setattr(settings, "embed_provider", "local")
    await _set_embedding_space(SAME_DIM_OTHER_MODEL, EMBEDDING_DIM, backfilled=True)
    with pytest.raises(RuntimeError, match="embedding model mismatch"):
        await verify_embedding_space()


async def test_refuses_a_model_of_the_wrong_dimension(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "local")
    monkeypatch.setattr(settings, "embed_model", WRONG_DIM_MODEL)
    with pytest.raises(RuntimeError, match="embedding dimension mismatch"):
        await verify_embedding_space()


async def test_warns_but_does_not_refuse_on_an_incomplete_backfill(monkeypatch, caplog):
    monkeypatch.setattr(settings, "embed_provider", "local")
    await _set_embedding_space(settings.embed_model, EMBEDDING_DIM, backfilled=False)
    with caplog.at_level(logging.WARNING, logger="haki.db"):
        await verify_embedding_space()  # must not raise
    assert "backfill INCOMPLETE" in caplog.text


async def test_silent_for_a_non_local_provider(monkeypatch):
    """The dev/CI default (HAKI_EMBED_PROVIDER=fake): nothing here to check,
    because nothing here produced the stored vectors."""
    monkeypatch.setattr(settings, "embed_provider", "fake")
    await _set_embedding_space(SAME_DIM_OTHER_MODEL, 1, backfilled=False)
    await verify_embedding_space()  # must not raise despite the garbage row
