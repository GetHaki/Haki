"""13 aout, LongMemEval diagnostic: does the SDK's own build_prompt_context
guard text (not the eval harness's answer_v3.txt -- the REAL product
surface customer integrations actually use) correctly handle a past-value
question when it is the ONLY instruction the model gets, same worst-case
methodology as scripts/probe_contested_conflict_serving.py?

Two contested-pair cases through the real SDK render: Wells Fargo
(current-value question, must still pick the newer fact -- Bug 3
regression check) and Apex Legends (past-value question, must now pick
the OLDER fact on purpose -- the new fix's target case).

Usage: uv run python scripts/probe_sdk_past_value_exception.py
Requires HAKI_LLM_API_KEY. Real cost: 2 questions x (answer+judge), a few
cents.
"""

import asyncio

from eval.env import llm_settings
from eval.llm import ChatClient
from eval.run import judge
from haki.runtime import build_prompt_context

CASES = [
    {
        "qid": "852ce960",
        "question": "What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?",
        "gold": "$400,000",
        "question_date": "2023/12/18 (Mon) 04:17",
        "predicate": "wells_fargo_pre_approval",
        "facts": [
            ({"amount": "$350,000"}, "2023-08-11T00:00:00+00:00"),
            ({"amount": "$400,000"}, "2023-11-30T00:00:00+00:00"),
        ],
    },
    {
        "qid": "9bbe84a2",
        "question": "What was my previous goal for my Apex Legends level before I updated my goal?",
        "gold": "level 100",
        "question_date": "2023/10/15",
        "predicate": "apex_legends_level_goal",
        "facts": [
            ({"goal": 100}, "2023-06-16T00:00:00+00:00"),
            ({"goal": 150}, "2023-09-30T00:00:00+00:00"),
        ],
    },
]


def build_packet(case: dict) -> dict:
    conflict_id = f"c-{case['qid']}"
    facts = []
    for i, (value, valid_from) in enumerate(case["facts"]):
        facts.append(
            {
                "id": f"f{i}",
                "predicate": case["predicate"],
                "value": value,
                "confidence": 0.9,
                "valid_from": valid_from,
                "source_event_ids": [f"evt-{i}"],
                "contested": True,
                "conflict_id": conflict_id,
            }
        )
    return {"facts": facts, "episodes": [], "warnings": []}


JUDGE_PROMPT = open("eval/prompts/judge_v1.txt", encoding="utf-8").read()


class _Q:
    pass


async def run_case(answer_client, judge_client, case: dict) -> bool:
    block = build_prompt_context(build_packet(case))
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the memory context below. Answer concisely (one short sentence or "
        "phrase).\n\n" + block
    )
    result = await answer_client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": case["question"]}]
    )
    answer = result.content

    q = _Q()
    q.qid = case["qid"]
    q.question = case["question"]
    q.answer = case["gold"]
    q.question_date = case["question_date"]
    q.qtype = "knowledge-update"
    q.abstention_expected = False
    verdict, _jpt, _jct = await judge(judge_client, JUDGE_PROMPT, q, answer)
    ok = verdict["label"] == "correct"
    print(
        f"  {case['qid']}: {'OK ' if ok else 'FAIL'} gold={case['gold']!r} "
        f"answer={answer!r} reason={verdict.get('judge_reason')!r}"
    )
    return ok


async def main() -> None:
    llm = llm_settings()
    answer_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    judge_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    print("=== SDK build_prompt_context guard text, as sole instruction ===")
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
