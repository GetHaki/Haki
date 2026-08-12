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
from eval.run import parse_judge_output


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
    assert [q.qtype for q in questions] == ["single-hop", "temporal", "adversarial", "open-domain"]
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
    questions = datasets.load_longmemeval(longmemeval_file)
    first = datasets.select(questions, subset=2)
    second = datasets.select(questions, subset=2)
    assert [q.qid for q in first] == ["ku_1", "tr_1"] == [q.qid for q in second]


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
