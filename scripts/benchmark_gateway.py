"""Gateway latency benchmark (sprint 7): the real cost of automatic memory.

Measures, against a LIVE server (uvicorn on :8100, real upstream provider):

- 10 calls WITH memory (X-Haki-Subject-Id): total client time + the
  X-Haki-Context-Ms header (server-side build_context time);
- 10 calls WITHOUT memory (no subject header): total client time — that is
  the pure pass-through path, i.e. ~the upstream latency itself.

The median difference (with - without) is the overhead Haki adds on top of
the provider call. LLM latency is noisy and dominates; run on a quiet
network. Usage:

    uv run python scripts/benchmark_gateway.py \
        --api-key hk_... --model openai/gpt-4o-mini
"""

import argparse
import statistics
import time

import httpx

GATEWAY_URL = "http://localhost:8100/gateway/v1/chat/completions"
MESSAGE = "Draft a one-line payment reminder for my invoice."


def run(api_key: str, model: str, calls: int, with_memory: bool) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    if with_memory:
        headers["X-Haki-Subject-Id"] = "usr_42"
        headers["X-Haki-Purpose"] = "benchmark"
    totals, context_ms = [], []
    with httpx.Client(timeout=120.0) as client:
        # One warm-up call outside the measurement (embedder, pools).
        client.post(
            GATEWAY_URL,
            json={"model": model, "messages": [{"role": "user", "content": MESSAGE}]},
            headers=headers,
        )
        for _ in range(calls):
            start = time.perf_counter()
            response = client.post(
                GATEWAY_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": MESSAGE}],
                },
                headers=headers,
            )
            totals.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
            if "x-haki-context-ms" in response.headers:
                context_ms.append(float(response.headers["x-haki-context-ms"]))
    return {"totals": totals, "context_ms": context_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()

    with_memory = run(args.api_key, args.model, args.calls, with_memory=True)
    without_memory = run(args.api_key, args.model, args.calls, with_memory=False)

    med_with = statistics.median(with_memory["totals"])
    med_without = statistics.median(without_memory["totals"])
    med_context = statistics.median(with_memory["context_ms"])

    print(f"calls per mode : {args.calls} (+ 1 warm-up each)")
    print(f"WITH memory    : median total {med_with:.0f} ms "
          f"(min {min(with_memory['totals']):.0f}, max {max(with_memory['totals']):.0f})")
    print(f"WITHOUT memory : median total {med_without:.0f} ms "
          f"(min {min(without_memory['totals']):.0f}, max {max(without_memory['totals']):.0f})")
    print(f"memory overhead (median, end-to-end) : {med_with - med_without:.0f} ms")
    print(f"build_context server-side (X-Haki-Context-Ms median) : {med_context:.1f} ms")


if __name__ == "__main__":
    main()
