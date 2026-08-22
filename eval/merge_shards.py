"""Merge N shard result files (eval.run --shard-index/--shard-count) into
one combined report, using the same report writer as a normal run so a
sharded cloud run and a single-process run produce byte-identical report
shapes.

Usage:
    uv run python -m eval.merge_shards --run-id <merged-id> \
        eval/results/longmemeval_s_gh-123-shard0.json \
        eval/results/longmemeval_s_gh-123-shard1.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import aggregate
from eval.report import write_reports

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded eval run results")
    parser.add_argument("shard_files", nargs="+", help="eval/results/*.json shard outputs")
    parser.add_argument("--run-id", required=True, help="run_id for the merged report")
    args = parser.parse_args()

    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.shard_files]
    if not shards:
        print("no shard files given")
        return 1

    dataset_names = {s["dataset"] for s in shards}
    if len(dataset_names) > 1:
        print(f"refusing to merge shards from different datasets: {dataset_names}")
        return 1

    # Shards of ONE run, or nothing. Merging shards drawn from different
    # samples produces a number that describes no experiment: the parts do
    # not add up to a whole, and the merged report would nonetheless look
    # exactly like a real one. Checked on the three things that define the
    # sample -- the file it was drawn from, the filter, and how it was
    # drawn.
    signatures = {
        json.dumps(
            {
                "sha256": shard.get("config", {}).get("dataset", {}).get("sha256"),
                "subset": shard.get("config", {}).get("selection", {}).get("subset"),
                "types": sorted(shard.get("config", {}).get("selection", {}).get("types") or []),
                "seed": shard.get("config", {}).get("selection", {}).get("seed"),
                "stratified": shard.get("config", {}).get("selection", {}).get("stratified"),
            },
            sort_keys=True,
        )
        for shard in shards
    }
    if len(signatures) > 1:
        print("refusing to merge shards drawn from different samples:")
        for signature in sorted(signatures):
            print(f"  {signature}")
        return 1

    records: list[dict] = []
    seen_qids: set[str] = set()
    for shard in shards:
        for record in shard["questions"]:
            if record["qid"] in seen_qids:
                # Shards are partitioned by history_id specifically to make
                # this impossible (see datasets.shard) — a duplicate here
                # means the shard files don't actually come from one
                # partitioned run, loud rather than silently double-counting.
                print(f"duplicate qid across shards: {record['qid']} — refusing to merge")
                return 1
            seen_qids.add(record["qid"])
            records.append(record)

    config = shards[0]["config"]
    composition: dict[str, int] = {}
    for record in records:
        qtype = record.get("qtype", "unknown")
        composition[qtype] = composition.get(qtype, 0) + 1
    config["selection"] = {
        **config.get("selection", {}),
        # Recomputed from the merged records rather than copied from a
        # shard: what the merged report describes is what it actually
        # contains.
        "composition": dict(sorted(composition.items())),
        "shard": f"merged from {len(shards)} shards",
    }
    summary = aggregate(records)
    json_path, md_path = write_reports(
        RESULTS_DIR, shards[0]["dataset"], args.run_id, config, records, summary
    )
    print(f"merged {len(records)} questions from {len(shards)} shards")
    print(f"rapport: {json_path}\n         {md_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
