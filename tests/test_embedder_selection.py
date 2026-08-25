"""Choosing an embedding model, and the two ways that choice fails silently.

The model was hardcoded until 22 aout. Making it configuration is only safe
if the two things a wrong value breaks are checked where they can still be
explained:

- the DIMENSION, which must match the `vector(N)` columns. Getting it wrong
  used to surface as a database error in the middle of a consolidation job;
- the PREFIXES, which a retrieval-trained model is trained with and which
  nothing else applies. fastembed's own `query_embed` does not (verified:
  cos(query_embed(q), embed(q)) == 1.0), so a question embedded as a passage
  just retrieves worse -- measured at 89.6 % against 87.4 % gold served with
  snowflake-arctic-embed-s, with no error anywhere.
"""

import pytest

from app.context import build_context
from app.db import async_session
from app.providers import EMBEDDING_DIM, get_embedder
from app.providers.base import Embedder
from app.providers.fake import FakeProvider
from app.providers.local import MODELS, LocalEmbedder

ASYMMETRIC = "intfloat/multilingual-e5-large"
DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _Recorder(LocalEmbedder):
    """A LocalEmbedder that records what would reach the model.

    Subclassing rather than mocking the module: the point is to exercise the
    real prefix/cache path, and only the ONNX call is replaced.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.seen: list[str] = []

    def _embed_missing(self, texts: list[str]) -> dict[str, list[float]]:
        self.seen.extend(texts)
        return {text: [0.0] * self._spec.dim for text in texts}


async def test_an_asymmetric_model_gets_its_two_prefixes():
    embedder = _Recorder(ASYMMETRIC)
    await embedder.embed_query(["when did she move"])
    await embedder.embed(["She moved to Lyon in March."])
    assert embedder.seen == [
        "query: when did she move",
        "passage: She moved to Lyon in March.",
    ]


async def test_the_default_model_is_symmetric_and_nothing_changed():
    """The split must be a no-op for the model every install already runs."""
    embedder = _Recorder(DEFAULT)
    await embedder.embed_query(["when did she move"])
    await embedder.embed(["when did she move"])
    assert embedder.seen == ["when did she move"]  # one call: same text, cached


async def test_an_unmeasured_model_is_refused_at_construction():
    with pytest.raises(RuntimeError, match="not a model this project has measured"):
        LocalEmbedder("nomic-ai/nomic-embed-text-v1.5")


async def test_a_model_of_the_wrong_dimension_is_refused_before_any_insert(monkeypatch):
    """The generalised form of the openai refusal.

    The message has to say that a swap needs a BACKFILL, not only a wider
    column: vectors from two models are not comparable, so migrating the
    column alone leaves every existing row with a vector that matches
    nothing -- a silently empty vector axis, which is this project's most
    expensive bug shape.
    """
    from app import providers

    monkeypatch.setattr(providers, "_embedder", None)
    monkeypatch.setattr(providers.settings, "embed_provider", "local")
    monkeypatch.setattr(providers.settings, "embed_model", DEFAULT)
    assert MODELS[DEFAULT].dim != EMBEDDING_DIM
    with pytest.raises(RuntimeError, match="re-embeds every stored row"):
        get_embedder()
    monkeypatch.setattr(providers, "_embedder", None)


async def test_build_context_embeds_the_query_as_a_query(client):
    """The wiring, not the model: which METHOD the hot path calls.

    Without this, adopting a retrieval-trained model would quietly keep
    embedding questions as passages and give back most of what it bought.
    """
    calls: list[str] = []

    class Watched(FakeProvider):
        async def embed(self, texts):
            calls.append("embed")
            return await super().embed(texts)

        async def embed_query(self, texts):
            calls.append("embed_query")
            return await FakeProvider.embed(self, texts)

    async with async_session() as session:
        await build_context(
            session,
            project_id="prj_support",
            subject_id="usr_no_such_subject",
            query="anything at all",
            budget_tokens=500,
            embedder=Watched(),
        )
        await session.commit()

    assert calls and calls[0] == "embed_query"


async def test_every_registered_model_satisfies_the_protocol():
    assert isinstance(LocalEmbedder(DEFAULT), Embedder)
    assert isinstance(FakeProvider(), Embedder)
