"""Context latency benchmark (sprint 3 — LA VITESSE).

Seeds N active facts (100 / 1 000 / 10 000) for one subject through the
Ledger (embeddings via the LOCAL fastembed provider), then runs 100
`build_context` calls (direct call, no HTTP) and reports p50/p95/p99 with a
breakdown: query embedding time vs SQL+ranking time.

Run: uv run python scripts/benchmark_context.py
Requires: Postgres up (docker compose), migrations applied to the main
`haki` database. Bench rows (org_id = org_bench) are wiped before and after.
"""

import asyncio
import statistics
import time
import uuid

from sqlalchemy import text

from app.context import build_context
from app.db import async_session, engine, install_tcp_nodelay
from app.ledger.core import create_fact
from app.models import FactStatus
from app.providers.local import LocalEmbedder

SIZES = [100, 1_000, 10_000]
N_QUERIES = 100
ORG = "org_bench"
PROJECT = "prj_bench"

# Pool of realistic French queries (repeats are intentional: they exercise
# the LRU query cache like real agent traffic does).
QUERIES = [
    "quelle langue pour les factures ?",
    "quel est le plan du client ?",
    "préférences de communication",
    "adresse de livraison du client",
    "contraint de budget du projet",
    "fréquence des rapports hebdomadaires",
    "fuseau horaire du client",
    "mode de paiement préféré",
    "contact principal côté client",
    "format des exports comptables",
    "relances automatiques activées ?",
    "langue des notifications",
    "seuil de validation des dépenses",
    "jours de disponibilité du support",
    "canal préféré pour les relances",
    "devise de facturation",
    "politique de remboursement",
    "niveau de détail des rapports",
    "destinataires des newsletters",
    "préférence de signature des documents",
]

_TOPICS = [
    ("invoice_language", "langue de facturation", "français"),
    ("plan", "plan tarifaire", "pro"),
    ("timezone", "fuseau horaire", "Europe/Paris"),
    ("payment", "mode de paiement", "virement"),
    ("report", "fréquence des rapports", "hebdomadaire"),
    ("channel", "canal de relance", "email"),
    ("currency", "devise", "EUR"),
    ("support", "disponibilité support", "jours ouvrés"),
    ("export", "format des exports", "csv"),
    ("budget", "seuil de validation", "500 euros"),
]


class TimingEmbedder:
    """Wraps an embedder and records per-call embedding latency (ms)."""

    def __init__(self, inner: LocalEmbedder) -> None:
        self.inner = inner
        self.calls_ms: list[float] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        vectors = await self.inner.embed(texts)
        self.calls_ms.append((time.perf_counter() - started) * 1000)
        return vectors


async def _wipe_bench_rows(session) -> None:
    await session.execute(text("DELETE FROM context_traces WHERE project_id = 'prj_bench'"))
    await session.execute(text("DELETE FROM facts WHERE org_id = 'org_bench'"))
    await session.commit()


async def seed(session, embedder: LocalEmbedder, n: int, subject_id: str) -> None:
    """Create n ACTIVE facts for the subject via the Ledger (local embeddings)."""
    texts = []
    for i in range(n):
        predicate, topic, value = _TOPICS[i % len(_TOPICS)]
        texts.append(f"{predicate}_{i} {topic} {value} variante {i}")
    embeddings: list[list[float]] = []
    batch = 512
    for start in range(0, n, batch):
        embeddings.extend(await embedder.embed(texts[start : start + batch]))

    for i in range(n):
        predicate, topic, value = _TOPICS[i % len(_TOPICS)]
        fact = await create_fact(
            session,
            org_id=ORG,
            project_id=PROJECT,
            subject_id=subject_id,
            predicate=f"{predicate}_{i}",
            value={"detail": f"{topic} {value} variante {i}"},
            confidence=0.9,
            source_event_ids=[uuid.uuid4()],
        )
        fact.embedding = embeddings[i]
        fact.search_text = texts[i]
        if (i + 1) % 1000 == 0:
            await session.commit()
    # Bulk promotion to active: seed setup, not business logic.
    await session.execute(
        text(
            "UPDATE facts SET status = :status "
            "WHERE org_id = :org AND subject_id = :subject"
        ),
        {"status": FactStatus.active.value, "org": ORG, "subject": subject_id},
    )
    await session.commit()


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    qs = statistics.quantiles(samples, n=100)
    return qs[49], qs[94], qs[98]


async def main() -> None:
    install_tcp_nodelay()
    embedder = LocalEmbedder()
    print("loading local embedding model (fastembed, first run downloads ~100 MB)...")
    t = time.perf_counter()
    await embedder.embed(["warmup"])
    print(f"model ready in {time.perf_counter() - t:.1f} s\n")

    async with async_session() as session:
        await _wipe_bench_rows(session)

    print(f"{'facts':>6} | {'p50':>7} {'p95':>7} {'p99':>7} | "
          f"{'embed p50':>9} {'sql p50':>8} | total=embed+sql+assemble (ms)")
    print("-" * 78)

    for n in SIZES:
        subject_id = f"bench_{n}"
        async with async_session() as session:
            t = time.perf_counter()
            await seed(session, embedder, n, subject_id)
            seed_s = time.perf_counter() - t

        timing = TimingEmbedder(embedder)
        totals: list[float] = []
        for i in range(N_QUERIES):
            query = QUERIES[i % len(QUERIES)]
            started = time.perf_counter()
            # One session per call, exactly like the API (get_session is
            # per-request): reusing one session would grow the identity map
            # and make every flush slower — an artifact, not the hot path.
            async with async_session() as session:
                await build_context(
                    session,
                    project_id=PROJECT,
                    subject_id=subject_id,
                    query=query,
                    budget_tokens=900,
                    embedder=timing,
                )
                await session.commit()
            totals.append((time.perf_counter() - started) * 1000)

        p50, p95, p99 = _percentiles(totals)
        e50, _, _ = _percentiles(timing.calls_ms)
        sql = [t_ - e for t_, e in zip(totals, timing.calls_ms)]
        s50, _, _ = _percentiles(sql)
        print(f"{n:>6} | {p50:7.1f} {p95:7.1f} {p99:7.1f} | "
              f"{e50:9.1f} {s50:8.1f} | (seed {seed_s:.1f} s)")

    async with async_session() as session:
        await _wipe_bench_rows(session)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
