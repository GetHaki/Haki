"""Bug 3 probe (11 aout oracle@900 finding, real cost, small): does the
chain-of-note answer prompt (eval/prompts/answer_v3.txt, untracked WIP
found this session) actually fix the temporal tie-break failure gpt-4o-mini
showed 3/3 on the original oracle test, or does it just look better on
paper?

Rebuilds the exact 3 named failing cases (5K personal best, Wells Fargo
pre-approval, bike count) from the REAL LongMemEval evidence sessions --
same dates, same values, quoted verbatim -- as a minimal two-fact memory
block (old + new value, exactly the shape a served packet can produce
once episodes carry raw historical text alongside a fact, see the 13 aout
"Bug 2" note on why this is not just a hypothetical). Runs BOTH answer_v2
(current default, known to fail 3/3) and answer_v3 (the candidate fix)
through the SAME real answer+judge pipeline, side by side.

Usage: uv run python scripts/probe_temporal_tiebreak.py
Requires HAKI_LLM_API_KEY in the environment/.env. Real cost: ~6 LLM
calls (3 questions x 2 prompts, answer+judge each), a few cents.
"""

import asyncio

from eval.env import llm_settings
from eval.llm import ChatClient
from eval.run import answer_with_memory, judge

CASES = [
    {
        "qid": "6a1eabeb",
        "question": "What was my personal best time in the charity 5K run?",
        "gold": "25 minutes and 50 seconds (or 25:50)",
        "question_date": "2023/06/25 (Sun) 13:22",
        "facts": [
            ("personal_best_5k", {"time": "27:12"}, "2023-05-23"),
            ("personal_best_5k", {"time": "25:50"}, "2023-05-30"),
        ],
    },
    {
        "qid": "852ce960",
        "question": "What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?",
        "gold": "$400,000",
        "question_date": "2023/12/18 (Mon) 04:17",
        "facts": [
            ("wells_fargo_pre_approval", {"amount": "$350,000"}, "2023-08-11"),
            ("wells_fargo_pre_approval", {"amount": "$400,000"}, "2023-11-30"),
        ],
    },
    {
        "qid": "89941a93",
        "question": "How many bikes do I currently own?",
        "gold": "4",
        "question_date": "2023/10/27 (Fri) 13:00",
        "facts": [
            ("bike_count", {"count": 3}, "2023-02-22"),
            ("bike_count", {"count": 4}, "2023-10-10"),
        ],
    },
]


def render_memory(facts: list[tuple[str, dict, str]]) -> str:
    lines = [f"- {p}: {v} (valid from {d})" for p, v, d in facts]
    return "Known facts about the user (from the memory system):\n" + "\n".join(lines)


class _Q:
    """Minimal stand-in for eval.datasets.Question (only the fields
    answer_with_memory/judge actually read)."""

    def __init__(self, qid, question, gold, question_date):
        self.qid = qid
        self.question = question
        self.answer = gold
        self.question_date = question_date
        self.qtype = "knowledge-update"
        self.abstention_expected = False


async def run_case(client, judge_client, prompt_name, prompt_text, case):
    q = _Q(case["qid"], case["question"], case["gold"], case["question_date"])
    memory = render_memory(case["facts"])
    answer, _pt, _ct = await answer_with_memory(client, prompt_text, q, memory)
    verdict, _jpt, _jct = await judge(judge_client, JUDGE_PROMPT, q, answer)
    ok = verdict["label"] == "correct"
    print(
        f"  [{prompt_name}] {case['qid']}: {'OK ' if ok else 'FAIL'} "
        f"answer={answer!r} verdict={verdict['label']} outdated={verdict.get('outdated')}"
    )
    return ok


JUDGE_PROMPT = open("eval/prompts/judge_v1.txt", encoding="utf-8").read()


async def main() -> None:
    llm = llm_settings()
    answer_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    judge_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")

    prompts = {
        "v2 (current)": open("eval/prompts/answer_v2.txt", encoding="utf-8").read(),
        "v3 (candidate)": open("eval/prompts/answer_v3.txt", encoding="utf-8").read(),
    }

    results: dict[str, list[bool]] = {}
    try:
        for name, text in prompts.items():
            print(f"\n=== {name} ===")
            oks = []
            for case in CASES:
                oks.append(await run_case(answer_client, judge_client, name, text, case))
            results[name] = oks
    finally:
        await answer_client.close()
        await judge_client.close()

    print("\n--- summary ---")
    for name, oks in results.items():
        print(f"{name}: {sum(oks)}/{len(oks)}")


if __name__ == "__main__":
    asyncio.run(main())
