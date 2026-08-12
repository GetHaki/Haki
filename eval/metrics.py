"""Metric aggregation for the eval harness.

Pure functions over per-question records (no I/O, no LLM): accuracy overall
and per question type, abstention accuracy, contradiction leakage, token
usage, context latency percentiles, cost. A record looks like:

{
    "qid": ..., "qtype": ..., "abstention_expected": bool,
    "systems": {
        "haki": {
            "label": "correct" | "incorrect" | "abstained",
            "outdated": bool,                # judge: relies on superseded info
            "context_tokens": int,           # packet tokens (haki) / answer prompt tokens (baseline)
            "latency_ms": float | None,
            "cost_usd": float,
        },
        "baseline": {...},
    },
}
"""

from __future__ import annotations

# Question types where "relies on outdated information" is the failure mode
# we track (LongMemEval knowledge-update: the history contains a value and
# its later replacement; answering with the old value is contradiction
# leakage).
LEAKAGE_TYPES = {"knowledge-update"}

LABELS = {"correct", "incorrect", "abstained"}


def is_success(label: str, abstention_expected: bool) -> bool:
    if label == "correct":
        return True
    # Proper abstention is a success ONLY when the gold answer says the
    # question is not answerable from the history.
    return label == "abstained" and abstention_expected


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(len(ordered) * pct / 100))
    return ordered[min(rank, len(ordered)) - 1]


def aggregate_system(records: list[dict], system: str) -> dict:
    judged = [r for r in records if system in r.get("systems", {})]
    n = len(judged)
    if n == 0:
        return {"n": 0}

    successes = 0
    per_type: dict[str, dict] = {}
    abstention_total = abstention_success = 0
    leakage_total = leakage_hits = 0
    context_tokens: list[float] = []
    latencies: list[float] = []
    cost = 0.0

    for record in judged:
        result = record["systems"][system]
        label = result.get("label", "incorrect")
        abstention_expected = bool(record.get("abstention_expected"))
        success = is_success(label, abstention_expected)
        successes += success

        bucket = per_type.setdefault(record["qtype"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += success

        if abstention_expected:
            abstention_total += 1
            abstention_success += label == "abstained"

        if record["qtype"] in LEAKAGE_TYPES:
            leakage_total += 1
            leakage_hits += bool(result.get("outdated"))

        if result.get("context_tokens") is not None:
            context_tokens.append(float(result["context_tokens"]))
        if result.get("latency_ms") is not None:
            latencies.append(float(result["latency_ms"]))
        cost += float(result.get("cost_usd", 0.0))

    for bucket in per_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["n"] if bucket["n"] else None

    return {
        "n": n,
        "accuracy": successes / n,
        "per_type": dict(sorted(per_type.items())),
        "abstention": {
            "n": abstention_total,
            "accuracy": (abstention_success / abstention_total) if abstention_total else None,
        },
        "contradiction_leakage": {
            "n": leakage_total,
            "rate": (leakage_hits / leakage_total) if leakage_total else None,
        },
        "context_tokens_mean": (sum(context_tokens) / len(context_tokens)) if context_tokens else None,
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "cost_usd": cost,
    }


def aggregate(records: list[dict]) -> dict:
    systems = sorted({s for r in records for s in r.get("systems", {})})
    return {system: aggregate_system(records, system) for system in systems}
