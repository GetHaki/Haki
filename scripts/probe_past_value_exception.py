"""13 aout, LongMemEval diagnostic (run 31705865474): does the "past value"
exception added to answer_v3.txt actually fix the real failing case
without breaking the original Bug 3 behavior it's built on top of?

Three real cases, exact values/dates from the actual runs:
  1. Apex Legends level goal (qid 9bbe84a2) -- question asks for the
     PAST value explicitly ("What was my previous goal... before I
     updated it?"). Was already answered CORRECTLY by gpt-4o-mini even
     with the old prompt (no explicit permission to prefer the earlier
     date) -- regression check: must stay correct now that the new
     prompt gives it explicit permission, not accidentally flip.
  2. Autographed baseball collection (qid 0ddfec37) -- question asks for
     the PAST value ("in the FIRST THREE MONTHS of collection"). Was
     answered WRONG (gave the current value, 35, instead of the
     requested historical one, 15) -- this is the case the fix targets.
  3. Wells Fargo pre-approval (Bug 3 original oracle, 11 aout) -- a
     CURRENT-value question, to confirm the past-value exception did not
     regress the original "always prefer most recent" behavior it sits
     alongside.

Usage: uv run python scripts/probe_past_value_exception.py
Requires HAKI_LLM_API_KEY. Real cost: 3 questions x (answer+judge),
gpt-4o-mini, a few cents.
"""

import asyncio

from eval.env import llm_settings
from eval.llm import ChatClient
from eval.run import answer_with_memory, judge

CASES = [
    {
        "qid": "9bbe84a2",
        "question": "What was my previous goal for my Apex Legends level before I updated my goal?",
        "gold": "level 100",
        "question_date": "2023/10/15",
        "facts": [
            ("apex_legends_level_goal", {"goal": 100}, "2023-06-16"),
            ("apex_legends_level_goal", {"goal": 150}, "2023-09-30"),
        ],
    },
    {
        "qid": "0ddfec37",
        "question": "How many autographed baseballs have I added to my collection in the first three months of collection?",
        "gold": "15",
        "question_date": "2024/01/10",
        "facts": [
            ("autographed_baseball_collection", {"count": 15}, "2023-07-11"),
            ("autographed_baseball_collection", {"count": 35}, "2023-12-30"),
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
]


def render_memory(facts: list[tuple[str, dict, str]]) -> str:
    lines = [f"- {p}: {v} (valid from {d})" for p, v, d in facts]
    return "Known facts about the user (from the memory system):\n" + "\n".join(lines)


class _Q:
    pass


JUDGE_PROMPT = open("eval/prompts/judge_v1.txt", encoding="utf-8").read()
ANSWER_V3_PROMPT = open("eval/prompts/answer_v3.txt", encoding="utf-8").read()


async def run_case(answer_client, judge_client, case):
    q = _Q()
    q.qid = case["qid"]
    q.question = case["question"]
    q.answer = case["gold"]
    q.question_date = case["question_date"]
    q.qtype = "knowledge-update"
    q.abstention_expected = False
    memory = render_memory(case["facts"])
    answer, _pt, _ct = await answer_with_memory(answer_client, ANSWER_V3_PROMPT, q, memory)
    verdict, _jpt, _jct = await judge(judge_client, JUDGE_PROMPT, q, answer)
    ok = verdict["label"] == "correct"
    print(
        f"  {case['qid']}: {'OK ' if ok else 'FAIL'} gold={case['gold']!r} "
        f"answer={answer!r} verdict={verdict['label']} reason={verdict.get('judge_reason')!r}"
    )
    return ok


async def main() -> None:
    llm = llm_settings()
    answer_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    judge_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    print("=== answer_v3.txt with the past-value exception ===")
    oks = []
    try:
        for case in CASES:
            oks.append(await run_case(answer_client, judge_client, case))
    finally:
        await answer_client.close()
        await judge_client.close()
    print(f"\n--- summary: {sum(oks)}/{len(oks)} ---")


if __name__ == "__main__":
    asyncio.run(main())
