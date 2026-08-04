from app.config import settings
from app.providers.base import (
    EMBEDDING_DIM,
    REJECT_REASONS,
    Embedder,
    ExtractedFact,
    Extractor,
    RawCandidate,
)
from app.providers.fake import FakeProvider

# Process-wide embedder singleton: the local ONNX model (~100 MB) is loaded
# once on first use, and the LRU query cache must survive across requests —
# recreating the embedder per request would put a ~2 s model load back into
# the context hot path (measured).
_embedder: Embedder | None = None


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

        _embedder = LocalEmbedder()
    elif settings.embed_provider == "openai":
        from app.providers.openai import OpenAIProvider

        _embedder = OpenAIProvider()
    elif settings.embed_provider == "fake":
        _embedder = FakeProvider()
    else:
        raise RuntimeError(
            f"Unknown HAKI_EMBED_PROVIDER {settings.embed_provider!r} "
            "(expected 'local', 'fake' or 'openai')"
        )
    return _embedder


__all__ = [
    "EMBEDDING_DIM",
    "REJECT_REASONS",
    "Embedder",
    "ExtractedFact",
    "Extractor",
    "FakeProvider",
    "RawCandidate",
    "get_embedder",
    "get_extractor",
]
