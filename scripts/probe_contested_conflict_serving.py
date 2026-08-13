"""13 aout, "stop hiding real conflicts" -- does serving a contested pair
through the REAL product surface (haki.runtime.build_prompt_context, not
the eval harness's own renderer) actually let gpt-4o-mini pick the correct
(most recent) value, using ONLY the SDK's own guard text as instruction?

Worst-case test: no customer system prompt helps here, no chain-of-note
answer prompt (eval/prompts/answer_v3.txt) is involved -- the block
produced by build_prompt_context is the ENTIRE instruction the model gets,
exactly the minimum a real integration could supply. If this fails, the
"serve instead of hide" change would be unsafe on its own and would need
to lean on a customer's own prompt to save it.

Same 3 real LongMemEval cases as scripts/probe_temporal_tiebreak.py (11
aout oracle@900 finding), rebuilt as a CONTESTED pair the way
app.context.build_context now actually produces one: two dated facts,
same predicate, `contested: True` with a shared `conflict_id`.

Usage: uv run python scripts/probe_contested_conflict_serving.py
Requires HAKI_LLM_API_KEY. Real cost: ~6 LLM calls (3 questions x
answer+judge), a few cents (checked OpenRouter balance ~$32 before
running).
"""

import asyncio

from eval.env import llm_settings
from eval.llm import ChatClient
from eval.run import judge
from haki.runtime import build_prompt_context

CASES = [
    {
        "qid": "6a1eabeb",
        "question": "What was my personal best time in the charity 5K run?",
        "gold": "25 minutes and 50 seconds (or 25:50)",
        "question_date": "2023/06/25 (Sun) 13:22",
        "predicate": "personal_best_5k",
        "facts": [
            ({"time": "27:12"}, "2023-05-23T00:00:00+00:00"),
            ({"time": "25:50"}, "2023-05-30T00:00:00+00:00"),
        ],
    },
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
        "qid": "89941a93",
        "question": "How many bikes do I currently own?",
        "gold": "4",
        "question_date": "2023/10/27 (Fri) 13:00",
        "predicate": "bike_count",
        "facts": [
            ({"count": 3}, "2023-02-22T00:00:00+00:00"),
            ({"count": 4}, "2023-10-10T00:00:00+00:00"),
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


async def run_case(answer_client: ChatClient, judge_client: ChatClient, case: dict) -> bool:
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

    class _Q:
        pass

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
        f"  {case['qid']}: {'OK ' if ok else 'FAIL'} answer={answer!r} "
        f"verdict={verdict['label']} outdated={verdict.get('outdated')}"
    )
    return ok


async def main() -> None:
    llm = llm_settings()
    answer_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    judge_client = ChatClient(llm["base_url"], llm["api_key"], "openai/gpt-4o-mini")
    print("=== contested pair served via build_prompt_context (SDK guard text only) ===")
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
