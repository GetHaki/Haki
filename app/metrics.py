"""In-memory operational counters — deliberately NOT a Prometheus-style
metrics system: no histograms, no labels cardinality, no scrape endpoint
format. Just a process-local dict of monotonic counters behind a lock,
enough to give the noisy-failure contract (ContextPacket.status,
X-Haki-Memory, MCP tool status) a cheap way to be OBSERVED over time
instead of only appearing in logs one request at a time.

Reset on process restart — this is visibility for "is this happening a
lot right now", not a durable audit trail (that is context_traces / the
structured logs already emitted alongside every degradation). Exposed at
GET /health under "counters".
"""

import threading

_lock = threading.Lock()
_counters: dict[str, int] = {}


def increment(name: str, by: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + by


def snapshot() -> dict[str, int]:
    """Point-in-time copy, safe to serialize (e.g. into a JSON response)."""
    with _lock:
        return dict(_counters)


def reset() -> None:
    """Test-only: clear all counters between tests."""
    with _lock:
        _counters.clear()
