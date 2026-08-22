"""Tests for the eval harness: dataset parsing, deterministic selection,
transcript truncation, metric aggregation, judge output parsing.

Hermetic: mini fabricated fixtures, no network, no LLM, no database access
(the shared conftest still provisions haki_test for the rest of the suite).
"""

import json
import subprocess
import sys

import pytest

from eval import datasets, metrics
from eval.run import (
    ROOT,
    merge_packets,
    mirror_subject,
    parse_judge_output,
    parse_mem0_judge_output,
    question_events_mirrored,
    target_speakers,
)


# --------------------------------------------------------------------------
# Fixtures: mini fabricated datasets (3-4 questions each)
# --------------------------------------------------------------------------

@pytest.fixture
def longmemeval_file(tmp_path):
    def session(sid, msgs):
        return [{"role": r, "content": c} for r, c in msgs]

    data = [
        {
            "question_id": "ku_1",
            "question_type": "knowledge-update",
            "question": "What car does the user drive now?",
            "answer": "Honda Civic",
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_session_ids": [["s1"], ["s2"]],
            "haystack_dates": ["2023/03/01 (Wed) 09:00", "2023/05/01 (Mon) 09:00"],
            "haystack_sessions": [
                session("s1", [("user", "I just bought a Toyota Corolla."), ("assistant", "Nice car!")]),
                session("s2", [("user", "I sold the Toyota and now drive a Honda Civic."), ("assistant", "Congrats!")]),
            ],
            "answer_session_ids": ["s2"],
        },
        {
            "question_id": "tr_1",
            "question_type": "temporal-reasoning",
            "question": "How long ago did the user start running?",
            "answer": "About 2 months ago",
            "question_date": "2023/06/15 (Thu) 10:00",
            "haystack_session_ids": [["s1"]],
            "haystack_dates": ["2023/04/15 (Sat) 08:00"],
            "haystack_sessions": [session("s1", [("user", "I started running today.")])],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "abs_1_abs",
            "question_type": "abstention",
            "question": "What is the user's favorite opera?",
            "answer": "This was never discussed.",
            "question_date": "2023/06/20 (Tue) 10:00",
            "haystack_session_ids": [["s1"]],
            "haystack_dates": ["2023/05/20 (Sat) 08:00"],
            "haystack_sessions": [session("s1", [("user", "I like jazz.")])],
            "answer_session_ids": [],
        },
        {
            "question_id": "ss_1",
            "question_type": "single-session-user",
            "question": "What instrument does the user play?",
            "answer": "Piano",
            "question_date": "2023/06/25 (Sun) 10:00",
            "haystack_session_ids": [["s1"]],
            "haystack_dates": ["2023/06/01 (Thu) 08:00"],
            "haystack_sessions": [session("s1", [("user", "I play piano every evening.")])],
            "answer_session_ids": ["s1"],
        },
    ]
    path = tmp_path / "mini_longmemeval.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def locomo_file(tmp_path):
    data = [
        {
            "sample_id": "mini_0",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a cat named Milo."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Cute!"},
                ],
                "session_2_date_time": "2:00 pm on 15 May, 2023",
                "session_2": [
                    {"speaker": "Alice", "dia_id": "D2:1", "text": "Milo was sick yesterday."},
                ],
            },
            "qa": [
                {"question": "What pet did Alice adopt?", "answer": "A cat", "category": 1, "evidence": ["D1:1"]},
                {"question": "When was Milo sick?", "answer": "14 May 2023", "category": 2, "evidence": ["D2:1"]},
                {"question": "What is Alice's dream job?", "answer": None, "category": 5, "evidence": []},
                {"question": "What animal is Milo?", "answer": "A cat", "category": 4, "evidence": ["D1:1", "D2:1"]},
            ],
        }
    ]
    path = tmp_path / "mini_locomo.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_longmemeval_parsing(longmemeval_file):
    questions = datasets.load_longmemeval(longmemeval_file)
    assert len(questions) == 4

    ku = questions[0]
    assert ku.qid == "ku_1"
    assert ku.qtype == "knowledge-update"
    assert ku.answer == "Honda Civic"
    assert not ku.abstention_expected
    assert len(ku.sessions) == 2
    # Dataset dates parsed, chronological, timezone-aware.
    assert ku.sessions[0].date < ku.sessions[1].date
    assert ku.sessions[0].date.tzinfo is not None
    assert ku.sessions[0].messages[0].speaker == "user"
    assert [s.session_id for s in ku.evidence_sessions] == ["s2"]

    abs_q = questions[2]
    assert abs_q.abstention_expected  # both via type and via _abs suffix
    assert abs_q.evidence_sessions == []


def test_locomo_parsing(locomo_file):
    questions = datasets.load_locomo(locomo_file)
    # 15 aout: category numbers -> names corrected against the official
    # LoCoMo eval script (category 1 = multi-hop, 4 = single-hop -- see
    # LOCOMO_CATEGORIES in eval/datasets.py).
    assert [q.qtype for q in questions] == ["multi-hop", "temporal", "adversarial", "single-hop"]
    assert [q.abstention_expected for q in questions] == [False, False, True, False]

    adversarial = questions[2]
    assert adversarial.answer == "unanswerable"

    multi_evidence = questions[3]
    assert {s.session_id for s in multi_evidence.evidence_sessions} == {"mini_0_s1", "mini_0_s2"}
    # Two sessions, dates parsed from the LoCoMo format.
    assert len(questions[0].sessions) == 2
    assert questions[0].sessions[0].date.month == 5
    assert questions[0].sessions[0].messages[0].speaker == "Alice"


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------

def test_select_subset_deterministic(longmemeval_file):
    """Same config in, same questions out -- in every process.

    The ids are no longer the first two in dataset order: since 21 Aug
    the sample is stratified by question type, so which two come back
    depends on the corpus's type mix rather than on file order (see
    tests/test_eval_sampling.py for the properties, and eval.datasets.select
    for why first-N was a bug). What this test guards is unchanged:
    repeating the call must repeat the sample exactly.
    """
    questions = datasets.load_longmemeval(longmemeval_file)
    first = datasets.select(questions, subset=2)
    second = datasets.select(questions, subset=2)
    assert [q.qid for q in first] == [q.qid for q in second]
    assert len(first) == 2


def test_select_can_reproduce_the_pre_stratification_behaviour(longmemeval_file):
    """`--no-stratify` exists to re-run an old number on purpose, and
    nothing else. If this ever stops returning dataset order, a
    pre-21-Aug result can no longer be reproduced at all."""
    questions = datasets.load_longmemeval(longmemeval_file)
    selected = datasets.select(questions, subset=2, stratify=False)
    assert [q.qid for q in selected] == ["ku_1", "tr_1"]


def test_select_types_filter_then_subset(longmemeval_file):
    questions = datasets.load_longmemeval(longmemeval_file)
    filtered = datasets.select(questions, types=["knowledge-update", "temporal-reasoning"])
    assert [q.qid for q in filtered] == ["ku_1", "tr_1"]
    assert [q.qid for q in datasets.select(questions, subset=1, types=["knowledge-update"])] == ["ku_1"]


# --------------------------------------------------------------------------
# Sharding (parallel cloud runs — GitHub Actions caps a job at 6h, far short
# of a full LoCoMo/LongMemEval run)
# --------------------------------------------------------------------------

def _q(qid, history_id=""):
    return datasets.Question(
        qid=qid,
        qtype="knowledge-update",
        question="q",
        answer="a",
        question_date=None,
        abstention_expected=False,
        sessions=[],
        history_id=history_id,
    )


def test_shard_count_one_is_a_no_op():
    questions = [_q("a"), _q("b"), _q("c")]
    assert datasets.shard(questions, 0, 1) == questions


def test_shard_partitions_without_gaps_or_overlaps():
    questions = [_q(str(i)) for i in range(11)]  # no history_id -> qid used
    shards = [datasets.shard(questions, i, 4) for i in range(4)]
    seen = [q.qid for s in shards for q in s]
    assert sorted(seen) == sorted(q.qid for q in questions)  # nothing lost
    assert len(seen) == len(set(seen))  # nothing duplicated across shards


def test_shard_never_splits_a_shared_history_locomo_style():
    """LoCoMo: ~200 questions can share ONE conversation (history_id). A
    naive index-based split would scatter them across shards, and since
    eval/run.py's ingestion cache is per-process, every shard touching that
    conversation would re-ingest it from scratch — real LLM cost multiplied
    by however many shards happen to touch it."""
    questions = (
        [_q(f"conv1_q{i}", history_id="conv1") for i in range(20)]
        + [_q(f"conv2_q{i}", history_id="conv2") for i in range(5)]
    )
    shards = [datasets.shard(questions, i, 3) for i in range(3)]
    for s in shards:
        history_ids = {q.history_id for q in s}
        assert len(history_ids) <= 1, f"shard mixes histories: {history_ids}"
    # every question still assigned to exactly one shard
    seen = [q.qid for s in shards for q in s]
    assert sorted(seen) == sorted(q.qid for q in questions)


# --------------------------------------------------------------------------
# Transcript truncation (baseline full-context)
# --------------------------------------------------------------------------

def test_truncate_sessions_keeps_most_recent():
    sessions = [
        datasets.Session(session_id=f"s{i}", date=datasets.FALLBACK_BASE,
                         messages=[datasets.Message("user", "x" * 400)])  # ~100 tokens each
        for i in range(5)
    ]
    kept, truncated = datasets.truncate_sessions(sessions, max_tokens=250)
    assert truncated
    assert [s.session_id for s in kept] == ["s3", "s4"]  # most recent, chronological order

    kept_all, truncated_all = datasets.truncate_sessions(sessions, max_tokens=10_000)
    assert not truncated_all and len(kept_all) == 5


# --------------------------------------------------------------------------
# Retrieval-need classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many Korean restaurants have I tried in my city?", "count"),
        ("How often do I attend yoga classes to help with my anxiety?", "count"),
        ("How much did the mortgage pre-approval amount change by?", "count"),
        ("Why did Caroline choose the adoption agency?", "narrative"),
        ("How does Melanie prioritize self-care?", "narrative"),
        (
            "Would Caroline still want to pursue counseling if she hadn't received support?",
            "narrative",
        ),
        ("Can you describe Melanie's morning routine?", "narrative"),
        ("What was my personal best time in the charity 5K run?", "point_value"),
        ("Where did Rachel move to after her recent relocation?", "point_value"),
        ("When did Caroline go to the LGBTQ support group?", "point_value"),
    ],
)
def test_classify_retrieval_need(question, expected):
    assert datasets.classify_retrieval_need(question) == expected


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _record(qid, qtype, label, *, abstention=False, outdated=False, tokens=100, cost=0.001, system="haki"):
    return {
        "qid": qid,
        "qtype": qtype,
        "abstention_expected": abstention,
        "systems": {
            system: {
                "label": label,
                "outdated": outdated,
                "context_tokens": tokens,
                "latency_ms": 50.0,
                "cost_usd": cost,
            }
        },
    }


def test_metrics_accuracy_per_type():
    records = [
        _record("q1", "knowledge-update", "correct"),
        _record("q2", "knowledge-update", "incorrect"),
        _record("q3", "temporal-reasoning", "correct"),
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["accuracy"] == pytest.approx(2 / 3)
    assert stats["per_type"]["knowledge-update"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert stats["per_type"]["temporal-reasoning"]["accuracy"] == 1.0


def test_metrics_abstention():
    records = [
        _record("q1", "abstention", "abstained", abstention=True),   # bonne abstention
        _record("q2", "abstention", "incorrect", abstention=True),   # hallucination
        _record("q3", "single-hop", "abstained"),                    # abstention indue -> échec
        _record("q4", "single-hop", "correct"),
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["abstention"] == {"n": 2, "accuracy": 0.5}
    assert stats["accuracy"] == pytest.approx(2 / 4)


def test_metrics_contradiction_leakage():
    records = [
        _record("q1", "knowledge-update", "correct", outdated=False),
        _record("q2", "knowledge-update", "incorrect", outdated=True),
        _record("q3", "knowledge-update", "incorrect", outdated=True),
        _record("q4", "temporal-reasoning", "incorrect", outdated=True),  # hors scope leakage
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["contradiction_leakage"] == {"n": 3, "rate": pytest.approx(2 / 3)}


def test_metrics_tokens_latency_cost():
    records = [
        _record("q1", "single-hop", "correct", tokens=500, cost=0.01),
        _record("q2", "single-hop", "correct", tokens=700, cost=0.02),
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["context_tokens_mean"] == pytest.approx(600)
    assert stats["latency_ms"]["p50"] == pytest.approx(50.0)
    assert stats["cost_usd"] == pytest.approx(0.03)


def test_metrics_cost_split_query_vs_ingest():
    """cost_ingest_usd (paid once per history) must not be counted as
    per-query cost -- a single combined cost_usd hides whether Haki is
    cheaper than a full-context baseline once ingestion amortizes."""
    records = [
        {
            "qid": "q1", "qtype": "single-hop", "abstention_expected": False,
            "systems": {"haki": {
                "label": "correct", "outdated": False, "context_tokens": 500,
                "latency_ms": 50.0, "cost_query_usd": 0.001, "cost_ingest_usd": 0.05,
            }},
        },
        {
            "qid": "q2", "qtype": "single-hop", "abstention_expected": False,
            "systems": {"haki": {
                "label": "correct", "outdated": False, "context_tokens": 500,
                "latency_ms": 50.0, "cost_query_usd": 0.001, "cost_ingest_usd": 0.0,
            }},
        },
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["cost_query_usd"] == pytest.approx(0.002)
    assert stats["cost_ingest_usd"] == pytest.approx(0.05)
    assert stats["n_ingested"] == 1
    assert stats["cost_per_query_usd"] == pytest.approx(0.001)
    assert stats["cost_per_ingest_usd"] == pytest.approx(0.05)
    assert stats["cost_usd"] == pytest.approx(0.0)  # old field absent on these records


def test_metrics_cost_split_falls_back_to_legacy_cost_usd():
    """Pre-split records (only cost_usd) must still aggregate -- treated as
    pure query cost, the dominant case for old runs/tests."""
    records = [_record("q1", "single-hop", "correct", cost=0.03)]
    stats = metrics.aggregate(records)["haki"]
    assert stats["cost_query_usd"] == pytest.approx(0.03)
    assert stats["cost_ingest_usd"] == pytest.approx(0.0)
    assert stats["cost_per_query_usd"] == pytest.approx(0.03)


def test_metrics_per_retrieval_need_breakdown():
    """Distinct from per_type (the dataset's own category) -- this is what
    KIND of memory representation the question needs, so a single
    combined accuracy doesn't hide that e.g. count questions fail
    differently than point-value ones. Records missing the field (older
    runs, before this classifier existed) land in their own "unknown"
    bucket instead of being dropped or crashing aggregation."""
    records = [
        {**_record("q1", "single-hop", "correct"), "retrieval_need": "point_value"},
        {**_record("q2", "single-hop", "incorrect"), "retrieval_need": "point_value"},
        {**_record("q3", "single-hop", "incorrect"), "retrieval_need": "count"},
        {**_record("q4", "single-hop", "correct")},  # no retrieval_need field
    ]
    stats = metrics.aggregate(records)["haki"]
    assert stats["per_retrieval_need"]["point_value"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert stats["per_retrieval_need"]["count"] == {"n": 1, "correct": 0, "accuracy": 0.0}
    assert stats["per_retrieval_need"]["unknown"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_metrics_two_systems_independent():
    records = [
        {**_record("q1", "single-hop", "correct"), },
        {**_record("q2", "single-hop", "incorrect"), },
    ]
    for record in records:
        record["systems"]["baseline"] = {
            "label": "correct", "outdated": False,
            "context_tokens": 50_000, "latency_ms": None, "cost_usd": 0.05,
        }
    summary = metrics.aggregate(records)
    assert summary["haki"]["accuracy"] == 0.5
    assert summary["baseline"]["accuracy"] == 1.0


# --------------------------------------------------------------------------
# Judge output parsing
# --------------------------------------------------------------------------

def test_parse_judge_output_clean_json():
    verdict = parse_judge_output(
        '{"label": "correct", "relies_on_outdated_information": false, "reason": "matches"}'
    )
    assert verdict == {"label": "correct", "outdated": False, "judge_reason": "matches"}


def test_parse_judge_output_with_prose_around():
    verdict = parse_judge_output(
        'Here is my verdict:\n{"label": "abstained", "relies_on_outdated_information": true, "reason": "old value"}'
    )
    assert verdict["label"] == "abstained"
    assert verdict["outdated"] is True


def test_parse_judge_output_garbage_is_incorrect():
    verdict = parse_judge_output("I cannot decide.")
    assert verdict["label"] == "incorrect"
    assert "unparseable" in verdict["judge_reason"]


def test_render_facts_empty():
    assert datasets.render_facts([]) == "(no facts in memory)"


# --------------------------------------------------------------------------
# Sprint 0 calibration (15 aout): mem0 protocol port
# --------------------------------------------------------------------------

def test_parse_mem0_judge_output_correct():
    verdict = parse_mem0_judge_output('{"label": "CORRECT"}')
    assert verdict["label"] == "correct"


def test_parse_mem0_judge_output_wrong():
    verdict = parse_mem0_judge_output(
        'The answer misses the date entirely.\n{"label": "WRONG"}'
    )
    assert verdict["label"] == "incorrect"


def test_parse_mem0_judge_output_garbage_is_incorrect():
    """Mirrors parse_judge_output's own fail-closed behavior -- an
    unparseable judge response must never silently count as a pass."""
    verdict = parse_mem0_judge_output("I cannot decide.")
    assert verdict["label"] == "incorrect"
    assert "unparseable" in verdict["judge_reason"]


def test_parse_mem0_judge_output_never_returns_abstained():
    """mem0's own protocol has no abstention concept (label is CORRECT or
    WRONG only) -- must never invent a third label calibration mode never
    actually produces."""
    for raw in ['{"label": "CORRECT"}', '{"label": "WRONG"}', "garbage"]:
        assert parse_mem0_judge_output(raw)["label"] in {"correct", "incorrect"}


def test_render_mem0_transcript_matches_mem0s_own_format():
    """Mirrors mem0's RAGManager.clean_chat_history exactly: one flat
    "{timestamp} | {speaker}: {text}" line per message, no session-boundary
    markers -- distinct from Haki's own render_transcript (used for every
    non-calibration baseline run), which keeps "--- session ... ---"
    headers."""
    sessions = [
        datasets.Session(
            session_id="s1",
            date=datasets.FALLBACK_BASE,
            messages=[
                datasets.Message(speaker="Alice", content="I adopted a cat."),
                datasets.Message(speaker="Bob", content="What's its name?"),
            ],
        )
    ]
    rendered = datasets.render_mem0_transcript(sessions)
    lines = rendered.split("\n")
    assert lines == [
        f"{datasets.FALLBACK_BASE.isoformat()} | Alice: I adopted a cat.",
        f"{datasets.FALLBACK_BASE.isoformat()} | Bob: What's its name?",
    ]
    # No session-boundary marker anywhere, unlike render_transcript.
    assert "---" not in rendered


async def test_answer_mem0_baseline_puts_question_before_context():
    """The mem0 protocol's own prompt is question-before-context -- kept
    even though it reads backwards, because reproducing the calibration
    target exactly is the entire point (research/
    Haki_Livre_Construction_2026-08-15.md, Sprint 0 step 2)."""
    from eval.llm import ChatResult
    from eval.run import answer_mem0_baseline

    class _FakeClient:
        def __init__(self):
            self.messages = None

        async def chat(self, messages, temperature=0.0, max_tokens=None, response_format=None):
            self.messages = messages
            return ChatResult(content="A cat", prompt_tokens=10, completion_tokens=2)

    system_prompt = (ROOT / "eval/prompts/mem0_baseline_system.txt").read_text(encoding="utf-8")
    user_template = (ROOT / "eval/prompts/mem0_baseline_user.txt").read_text(encoding="utf-8")
    client = _FakeClient()
    answer, ptok, ctok = await answer_mem0_baseline(
        client, system_prompt, user_template, "What pet did Alice adopt?", "Alice: I adopted a cat."
    )
    assert answer == "A cat"
    assert client.messages[0]["role"] == "system"
    assert client.messages[1]["role"] == "user"
    body = client.messages[1]["content"]
    assert body.index("What pet did Alice adopt?") < body.index("I adopted a cat.")
    assert "# Question:" in body and "# Context:" in body and "# Short answer:" in body


async def test_judge_mem0_uses_json_response_format_and_no_extra_injection():
    """mem0's ACCURACY_PROMPT is a single user message with no per-qtype
    extra text -- distinct from Haki's own judge(), which injects
    knowledge-update/abstention notes (13-14 aout fixes). Injecting those
    here would stop this from being the same instrument as the calibration
    target."""
    from eval.llm import ChatResult
    from eval.run import judge_mem0

    class _FakeClient:
        def __init__(self):
            self.messages = None
            self.response_format = None

        async def chat(self, messages, temperature=0.0, max_tokens=None, response_format=None):
            self.messages = messages
            self.response_format = response_format
            return ChatResult(content='{"label": "CORRECT"}', prompt_tokens=20, completion_tokens=3)

    judge_prompt = (ROOT / "eval/prompts/mem0_judge.txt").read_text(encoding="utf-8")
    question = datasets.Question(
        qid="q1", qtype="knowledge-update", question="What car do I drive?",
        answer="Honda Civic", question_date=None, abstention_expected=False, sessions=[],
    )
    client = _FakeClient()
    verdict, ptok, ctok = await judge_mem0(client, judge_prompt, question, "A Honda Civic")
    assert verdict["label"] == "correct"
    assert client.response_format == {"type": "json_object"}
    assert len(client.messages) == 1 and client.messages[0]["role"] == "user"
    body = client.messages[0]["content"]
    assert "What car do I drive?" in body
    assert "Honda Civic" in body
    # No qtype-specific note injected (unlike Haki's own judge()).
    assert "knowledge UPDATE" not in body


# --------------------------------------------------------------------------
# Sprint 0 calibration (15 aout): LongMemEval official protocol port
# --------------------------------------------------------------------------

def test_render_lme_session_history_matches_official_format():
    """Mirrors run_generation.py's prepare_prompt exactly for
    retriever_type="orig-session", history_format="json": one
    '### Session N:' block per session with a JSON-dumped [{"role",
    "content"}, ...] turn list, sessions in chronological order."""
    sessions = [
        datasets.Session(
            session_id="s1",
            date=datasets.FALLBACK_BASE,
            messages=[datasets.Message(speaker="user", content="I adopted a cat.")],
        )
    ]
    rendered = datasets.render_lme_session_history(sessions)
    assert "### Session 1:" in rendered
    assert f"Session Date: {datasets.FALLBACK_BASE.isoformat()}" in rendered
    assert json.dumps([{"role": "user", "content": "I adopted a cat."}]) in rendered


def test_judge_lme_templates_cover_every_real_qtype():
    """Every qtype Haki's own LongMemEval loader actually produces must
    have a dispatch entry, or judge_lme silently falls back to the
    "standard" template for a type that needs different grading (e.g.
    temporal-reasoning's off-by-one forgiveness)."""
    from eval.run import LME_JUDGE_TEMPLATES

    real_qtypes = {
        "single-session-user", "single-session-assistant", "single-session-preference",
        "multi-session", "temporal-reasoning", "knowledge-update",
    }
    assert real_qtypes <= LME_JUDGE_TEMPLATES.keys()


async def test_judge_lme_dispatches_temporal_template_and_forgives_off_by_one():
    """The temporal-reasoning template is the one place the official
    protocol explicitly forgives off-by-one day errors -- picking the
    wrong template here would silently make Haki's temporal accuracy look
    worse than the calibration target intends."""
    from eval.llm import ChatResult
    from eval.run import judge_lme

    class _FakeClient:
        def __init__(self):
            self.messages = None

        async def chat(self, messages, temperature=0.0, max_tokens=None, response_format=None):
            self.messages = messages
            return ChatResult(content="yes", prompt_tokens=15, completion_tokens=1)

    question = datasets.Question(
        qid="tr_1", qtype="temporal-reasoning", question="How many days ago?",
        answer="18 days", question_date=None, abstention_expected=False, sessions=[],
    )
    client = _FakeClient()
    verdict, ptok, ctok = await judge_lme(client, question, "19 days")
    assert verdict["label"] == "correct"
    assert "off-by-one" in client.messages[0]["content"]


async def test_answer_lme_baseline_truncates_oversized_history():
    """15 aout: a real LongMemEval-S haystack that overflows gpt-4o-mini's
    context window must be truncated BEFORE the API call, not left to fail
    as a 400 -- mirrors run_generation.py's own
    `max_retrieval_length = model_max_length - gen_length - 1000` truncation
    (kept even though it keeps the EARLIEST tokens and drops the most
    recent ones, an odd choice in the source -- fidelity to the
    calibration target, not a "fix")."""
    from eval.llm import ChatResult
    from eval.run import LME_MAX_RETRIEVAL_TOKENS, answer_lme_baseline

    class _FakeClient:
        def __init__(self):
            self.messages = None

        async def chat(self, messages, temperature=0.0, max_tokens=None, response_format=None):
            self.messages = messages
            return ChatResult(content="answer", prompt_tokens=10, completion_tokens=1)

    oversized_history = "x" * (LME_MAX_RETRIEVAL_TOKENS * datasets.CHARS_PER_TOKEN * 2)
    client = _FakeClient()
    await answer_lme_baseline(client, "{history}", oversized_history, "2023/01/01", "q?")
    sent_history_len = len(client.messages[0]["content"])
    assert sent_history_len <= LME_MAX_RETRIEVAL_TOKENS * datasets.CHARS_PER_TOKEN


async def test_judge_lme_dispatches_abstention_template_on_abs_suffix():
    """Mirrors the official harness's own dispatch rule verbatim:
    `abstention='_abs' in entry['question_id']` -- overrides the qtype
    template regardless of what qtype the question otherwise carries."""
    from eval.llm import ChatResult
    from eval.run import judge_lme

    class _FakeClient:
        def __init__(self):
            self.messages = None

        async def chat(self, messages, temperature=0.0, max_tokens=None, response_format=None):
            self.messages = messages
            return ChatResult(content="yes", prompt_tokens=15, completion_tokens=1)

    question = datasets.Question(
        qid="abs_1_abs", qtype="abstention", question="What is my favorite opera?",
        answer="This was never discussed.", question_date=None,
        abstention_expected=True, sessions=[],
    )
    client = _FakeClient()
    verdict, ptok, ctok = await judge_lme(client, question, "I don't have that information.")
    assert verdict["label"] == "correct"
    assert "unanswerable" in client.messages[0]["content"]


# --------------------------------------------------------------------------
# Shard merging (eval.merge_shards — combines parallel cloud shards back
# into one report; real subprocess, same pattern as tests/test_cli.py and
# tests/test_mcp.py for exercising an actual CLI entrypoint)
# --------------------------------------------------------------------------

def _shard_file(tmp_path, name, dataset, run_id, records):
    from eval.report import write_reports

    path = write_reports(
        tmp_path,
        dataset,
        run_id,
        {"name": dataset, "dataset": {}, "selection": {}},
        records,
        metrics.aggregate(records),
    )[0]
    renamed = tmp_path / name
    path.rename(renamed)
    return renamed


def _run_merge(tmp_path, *shard_paths, run_id="merged-test"):
    return subprocess.run(
        [sys.executable, "-m", "eval.merge_shards", "--run-id", run_id, *map(str, shard_paths)],
        capture_output=True,
        text=True,
    )


def test_merge_shards_combines_records_and_recomputes_summary(tmp_path):
    shard0 = _shard_file(
        tmp_path, "s0.json", "toy", "run-shard0",
        [_record("q1", "knowledge-update", "correct")],
    )
    shard1 = _shard_file(
        tmp_path, "s1.json", "toy", "run-shard1",
        [_record("q2", "knowledge-update", "incorrect")],
    )
    from eval.run import RESULTS_DIR

    run_id = "test-merge-combine"
    try:
        result = _run_merge(tmp_path, shard0, shard1, run_id=run_id)
        assert result.returncode == 0, result.stdout + result.stderr
        merged = json.loads((RESULTS_DIR / f"toy_{run_id}.json").read_text(encoding="utf-8"))
        assert {r["qid"] for r in merged["questions"]} == {"q1", "q2"}
        assert merged["summary"]["haki"]["accuracy"] == 0.5
    finally:
        (RESULTS_DIR / f"toy_{run_id}.json").unlink(missing_ok=True)
        (RESULTS_DIR / f"toy_{run_id}.md").unlink(missing_ok=True)


def test_merge_shards_refuses_duplicate_qid_across_shards(tmp_path):
    # Shards are partitioned by history_id specifically so this can't
    # legitimately happen (see datasets.shard) — a duplicate means the
    # inputs aren't really disjoint shards, and merging would silently
    # double-count a question instead of catching the mistake.
    shard0 = _shard_file(
        tmp_path, "s0.json", "toy", "run-shard0",
        [_record("dup", "knowledge-update", "correct")],
    )
    shard1 = _shard_file(
        tmp_path, "s1.json", "toy", "run-shard1",
        [_record("dup", "knowledge-update", "incorrect")],
    )
    result = _run_merge(tmp_path, shard0, shard1, run_id="test-merge-dup")
    assert result.returncode != 0
    assert "duplicate qid" in result.stdout


def test_merge_shards_refuses_different_datasets(tmp_path):
    shard0 = _shard_file(
        tmp_path, "s0.json", "dataset_a", "run-shard0",
        [_record("q1", "knowledge-update", "correct")],
    )
    shard1 = _shard_file(
        tmp_path, "s1.json", "dataset_b", "run-shard1",
        [_record("q2", "knowledge-update", "correct")],
    )
    result = _run_merge(tmp_path, shard0, shard1, run_id="test-merge-mismatch")
    assert result.returncode != 0
    assert "different datasets" in result.stdout


# --------------------------------------------------------------------------
# Mechanism E4 (15 aout, Sprint 1): mirrored per-speaker stores
# --------------------------------------------------------------------------

def _speaker_question(question="what did Caroline say?", speakers=("Caroline", "Melanie")):
    return datasets.Question(
        qid="conv1_q0",
        qtype="single-hop",
        question=question,
        answer="a",
        question_date=None,
        abstention_expected=False,
        sessions=[
            datasets.Session(
                session_id="conv1_s1",
                date=datasets.FALLBACK_BASE,
                messages=[
                    datasets.Message(speaker="Caroline", content="I moved from Sweden."),
                    datasets.Message(speaker="Melanie", content="I went camping."),
                ],
            )
        ],
        history_id="conv1",
        speakers=list(speakers),
    )


def test_mirror_subject_is_scoped_per_speaker():
    assert mirror_subject("conv1", "Caroline") == "conv1__Caroline"
    assert mirror_subject("conv1", "Melanie") == "conv1__Melanie"


def test_question_events_mirrored_normalizes_role_and_keeps_the_real_speaker_in_text():
    question = _speaker_question()
    events = question_events_mirrored(question, "Caroline", "org", "prj", "run1")
    assert len(events) == 1
    assert events[0]["subject_id"] == "conv1__Caroline"
    messages = events[0]["payload"]["messages"]
    caroline_msg = next(m for m in messages if m["content"].startswith("Caroline:"))
    melanie_msg = next(m for m in messages if m["content"].startswith("Melanie:"))
    assert caroline_msg["role"] == "user"
    assert caroline_msg["content"] == "Caroline: I moved from Sweden."
    assert melanie_msg["role"] == "assistant"
    assert melanie_msg["content"] == "Melanie: I went camping."


def test_target_speakers_routes_to_the_one_named_in_the_question():
    question = _speaker_question(question="Where did Caroline move from?")
    assert target_speakers(question) == ["Caroline"]


def test_target_speakers_queries_both_when_neither_or_both_are_named():
    both_named = _speaker_question(question="Did Caroline or Melanie go camping?")
    assert target_speakers(both_named) == ["Caroline", "Melanie"]
    neither_named = _speaker_question(question="Who went camping?")
    assert target_speakers(neither_named) == ["Caroline", "Melanie"]


def test_merge_packets_passes_a_single_body_through_untouched():
    body = {"packet": {"facts": [{"id": "f1"}], "episodes": []}, "token_count": 10}
    assert merge_packets([body]) is body


def test_merge_packets_dedupes_and_sums_tokens_across_mirror_stores():
    body_a = {
        "packet": {
            "facts": [{"id": "f1", "predicate": "born_in", "value": {"place": "Sweden"}}],
            "episodes": [{"event_id": "e1", "excerpt": "..."}],
        },
        "token_count": 30,
        "trace_id": "trace-a",
    }
    body_b = {
        "packet": {
            "facts": [{"id": "f2", "predicate": "hobby", "value": {"name": "camping"}}],
            "episodes": [{"event_id": "e1", "excerpt": "..."}],  # same event, both stores
        },
        "token_count": 25,
        "trace_id": "trace-b",
    }
    merged = merge_packets([body_a, body_b])
    assert {f["id"] for f in merged["packet"]["facts"]} == {"f1", "f2"}
    assert [e["event_id"] for e in merged["packet"]["episodes"]] == ["e1"]
    assert merged["token_count"] == 55
    assert merged["trace_ids"] == ["trace-a", "trace-b"]
