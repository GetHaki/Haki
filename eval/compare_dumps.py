"""Compare two `retrieval_bench --dump` files PAIRED, question by question.

    uv run python -m eval.compare_dumps before.jsonl after.jsonl

Why this exists. The retrieval bench reports a percentage, and a percentage
gap invites the wrong reading: on n=231 a 2-point move is 5 questions, and
this project has already drawn -- and then withdrawn -- several A/B
conclusions from gaps that size. Two runs of the bench are PAIRED (the same
231 questions, the same evidence annotations), so the right test is McNemar
on the discordant pairs: of the questions whose outcome changed, how
lopsided is the change? Questions both runs got right, or both got wrong,
carry no information about which variant is better and are excluded by
construction.

The exact binomial two-sided p-value is computed here rather than the
chi-square approximation, because the discordant count is routinely under
25 (where the approximation is not trustworthy) and the exact test costs
nothing at this size.

Reported per metric (`any` and `complete`) and per LoCoMo category. A
category line is descriptive only: with n=11 on open-domain no test has
the power to say anything, and the output says so rather than printing a
p-value that invites over-reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

METRICS = ("any", "complete")
# Below this many discordant pairs, a p-value is noise dressed as evidence.
_MIN_DISCORDANT_FOR_A_TEST = 6


def _load(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        raise SystemExit(f"missing dump: {path}")
    records: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["sample_id"]), str(record["question"]))
        if key in records:
            # Same question text twice in one conversation: keep the first
            # and say so, rather than silently pairing the wrong two.
            print(f"  warning: duplicate question, second ignored: {key[1][:60]}")
            continue
        records[key] = record
    return records


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts (b, c).

    Under H0 each discordant pair is a fair coin, so b ~ Binomial(b+c, 1/2).
    The two-sided p is twice the smaller tail, clamped at 1.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _line(label: str, before: list[bool], after: list[bool], testable: bool) -> None:
    b = sum(1 for x, y in zip(before, after) if x and not y)  # after lost it
    c = sum(1 for x, y in zip(before, after) if y and not x)  # after won it
    n = len(before)
    rate_before = 100 * sum(before) / n
    rate_after = 100 * sum(after) / n
    verdict = ""
    if testable:
        if b + c < _MIN_DISCORDANT_FOR_A_TEST:
            verdict = f"  {b + c} discordant -- too few to test"
        else:
            p = mcnemar_exact(b, c)
            verdict = f"  p = {p:.4f}" + (
                "  SIGNIFICANT" if p < 0.05 else "  not significant"
            )
    print(
        f"  {label:<14} {rate_before:6.1f}% -> {rate_after:6.1f}%  "
        f"({rate_after - rate_before:+5.1f})   won {c:3d}  lost {b:3d}{verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    shared = sorted(before.keys() & after.keys())
    if not shared:
        print("the two dumps share no question -- different datasets?", file=sys.stderr)
        return 2
    dropped = (len(before) - len(shared)) + (len(after) - len(shared))
    print(
        f"\npaired on {len(shared)} questions"
        + (f"  ({dropped} unpaired record(s) ignored)" if dropped else "")
    )
    print(f"  before: {args.before}\n  after:  {args.after}\n")

    for metric in METRICS:
        print(f"metric: {metric}")
        _line(
            "OVERALL",
            [bool(before[k][metric]) for k in shared],
            [bool(after[k][metric]) for k in shared],
            testable=True,
        )
        by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in shared:
            by_category[str(before[key]["category"])].append(key)
        for category, keys in sorted(by_category.items()):
            _line(
                f"  {category}",
                [bool(before[k][metric]) for k in keys],
                [bool(after[k][metric]) for k in keys],
                # Per-category n is 11 to 114 here: report the movement,
                # do not put a p-value on a slice this thin.
                testable=False,
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
