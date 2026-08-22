from app.config import settings
from app.providers.base import (
    EMBEDDING_DIM,
    REJECT_REASONS,
    Embedder,
    ExtractedFact,
    Extractor,
    RawCandidate,
    Reranker,
)
from app.providers.fake import FakeProvider

# Process-wide embedder singleton: the local ONNX model (~100 MB) is loaded
# once on first use, and the LRU query cache must survive across requests —
# recreating the embedder per request would put a ~2 s model load back into
# the context hot path (measured).
_embedder: Embedder | None = None
# Process-wide reranker singleton (mechanism F-R): same reasoning as the
# embedder above -- the ONNX cross-encoder session must be loaded once, not
# per request.
_reranker: Reranker | None = None


def get_extractor() -> Extractor:
    """Select the extraction provider (HAKI_LLM_PROVIDER=fake|openai).

    Extraction runs in the consolidator (async worker), never in the context
    hot path, so a remote LLM is fine here. Default: fake.
    """
    if settings.llm_provider == "openai":
        from app.providers.openai import OpenAIProvider

        return OpenAIProvider()
    if settings.llm_provider == "fake":
        return FakeProvider()
    raise RuntimeError(
        f"Unknown HAKI_LLM_PROVIDER {settings.llm_provider!r} (expected 'fake' or 'openai')"
    )


def get_embedder() -> Embedder:
    """Select the embedding provider (HAKI_EMBED_PROVIDER=local|fake|openai).

    The embedder sits in the `POST /v1/context` hot path (query embedding),
    so the default is LOCAL (fastembed, ONNX CPU): no network call per query.
    The instance is a process-wide singleton (see module note).
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    if settings.embed_provider == "local":
        from app.providers.local import LocalEmbedder

        embedder = LocalEmbedder()
        # Same refusal as the openai branch below, generalised (22 aout):
        # the model is configuration now, so a model whose vectors do not
        # fit the `vector(N)` columns has to fail HERE and say why, not at
        # the first INSERT of a consolidation job. Changing the model is a
        # migration -- every stored embedding was produced by the previous
        # one and none of them are comparable to the new one.
        if embedder.spec.dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"HAKI_EMBED_MODEL={settings.embed_model!r} produces "
                f"{embedder.spec.dim}-dimensional vectors and the embedding "
                f"columns are vector({EMBEDDING_DIM}). Adopting it needs a "
                "migration that widens those columns AND re-embeds every "
                "stored row: vectors from two different models cannot be "
                "compared, so a swap without a backfill silently empties the "
                "vector axis for all existing data."
            )
        _embedder = embedder
    elif settings.embed_provider == "openai":
        # Refused, loudly, rather than accepted and failed at INSERT time
        # (22 aout). text-embedding-3-small returns 1536 dimensions and
        # `facts.embedding` is vector(384) since migration 0003, so this
        # selection has never been able to work -- it was documented in a
        # docstring nobody reads before setting an environment variable,
        # and the failure surfaced as a database error in the middle of a
        # consolidation job.
        raise RuntimeError(
            "HAKI_EMBED_PROVIDER=openai is not supported: this provider "
            "returns 1536-dimensional vectors and facts.embedding is "
            "vector(384) (migration 0003). Use HAKI_EMBED_PROVIDER=local. "
            "OpenAI remains available as an EXTRACTOR "
            "(HAKI_LLM_PROVIDER=openai)."
        )
    elif settings.embed_provider == "fake":
        _embedder = FakeProvider()
    else:
        raise RuntimeError(
            f"Unknown HAKI_EMBED_PROVIDER {settings.embed_provider!r} "
            "(expected 'local', 'fake' or 'openai')"
        )
    return _embedder


def get_reranker() -> Reranker:
    """Select the reranker provider (HAKI_RERANK_PROVIDER=local|fake).

    Only consulted when HAKI_RERANK_ENABLED is set -- callers check that
    flag themselves (see app.context) rather than this function silently
    returning a no-op, so "reranking is on" is never ambiguous between
    "disabled" and "enabled with a null provider". Process-wide singleton
    (see module note on _embedder).
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    if settings.rerank_provider == "local":
        from app.providers.local import LocalReranker

        _reranker = LocalReranker()
    elif settings.rerank_provider == "fake":
        _reranker = FakeProvider()
    else:
        raise RuntimeError(
            f"Unknown HAKI_RERANK_PROVIDER {settings.rerank_provider!r} "
            "(expected 'local' or 'fake')"
        )
    return _reranker


__all__ = [
    "EMBEDDING_DIM",
    "REJECT_REASONS",
    "Embedder",
    "ExtractedFact",
    "Extractor",
    "FakeProvider",
    "RawCandidate",
    "Reranker",
    "get_embedder",
    "get_extractor",
    "get_reranker",
]
