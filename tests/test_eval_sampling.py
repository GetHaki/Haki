"""Question sampling: what makes two eval runs comparable.

Until 21 Aug `select` took the first N questions in dataset order. On
LoCoMo, dataset order is conversation order, so a 180-question run and a
458-question run sampled different conversations AND different type mixes
-- and this project published a trajectory (17.1 % -> 30.6 % -> 31.4 %)
built from exactly those incomparable samples.

These tests pin the three properties that fix it: the sample keeps the
corpus's composition, it is the same in every process, and a different
seed gives an independent sample of the same shape.
"""

from dataclasses import replace

import pytest

from eval.datasets import DEFAULT_SEED, Question, composition, select

# A corpus with a deliberately skewed type mix, like LoCoMo's own
# (single-hop 55 %, temporal 21 %, multi-hop 18 %, open-domain 6 %).
CORPUS = [
    Question(
        qid=f"{qtype}-{i}",
        qtype=qtype,
        question="q",
        answer="a",
        question_date=None,
        abstention_expected=False,
        sessions=[],
        history_id=f"h{i}",
    )
    for qtype, count in (
        ("single-hop", 550),
        ("temporal", 210),
        ("multi-hop", 180),
        ("open-domain", 60),
    )
    for i in range(count)
]


def _share(questions: list[Question], qtype: str) -> float:
    return composition(questions).get(qtype, 0) / len(questions)


def test_the_sample_keeps_the_corpus_composition():
    sample = select(CORPUS, subset=200)
    assert len(sample) == 200
    for qtype in composition(CORPUS):
        assert _share(sample, qtype) == pytest.approx(_share(CORPUS, qtype), abs=0.01)


def test_the_old_first_n_behaviour_did_not():
    """The regression, stated as a measurement.

    Kept as a test rather than a comment: it is the reason the default
    changed, and if it ever stops being true the change stops being
    justified.
    """
    old = select(CORPUS, subset=200, stratify=False)
    assert _share(old, "single-hop") == 1.0
    assert _share(old, "temporal") == 0.0


def test_the_sample_is_the_same_in_every_process():
    """No Python hash() anywhere in the ordering.

    hash() is randomised per interpreter unless PYTHONHASHSEED is pinned,
    so an ordering built on it reshuffles the sample between runs -- which
    is precisely what this function exists to prevent. sha1 of
    "seed:qid" cannot do that, and the assertion below is on the exact
    ids, not just their count.
    """
    first = [q.qid for q in select(CORPUS, subset=137)]
    second = [q.qid for q in select(CORPUS, subset=137)]
    assert first == second
    assert first == [q.qid for q in select(CORPUS, subset=137, seed=DEFAULT_SEED)]


def test_another_seed_gives_an_independent_sample_of_the_same_shape():
    """What makes a variance estimate possible at all.

    One sample gives one number with no error bar. Several seeds, same
    composition, give a spread -- the only honest way to say whether a
    two-point difference between two runs means anything.
    """
    a = select(CORPUS, subset=200, seed=1)
    b = select(CORPUS, subset=200, seed=2)
    assert composition(a) == composition(b)
    assert [q.qid for q in a] != [q.qid for q in b]
    overlap = len({q.qid for q in a} & {q.qid for q in b}) / 200
    assert overlap < 0.5, "two seeds should not be drawing nearly the same questions"


def test_the_quotas_add_up_exactly():
    """Largest-remainder apportionment, checked on the awkward sizes.

    Naive rounding either overshoots the requested size or silently drops
    a small stratum whose share rounds to zero -- open-domain is 6 % of
    this corpus, so at N=17 its exact quota is 1.02 and at N=13 it is
    0.78.
    """
    for n in (7, 13, 17, 99, 101, 999):
        sample = select(CORPUS, subset=n)
        assert len(sample) == n, f"asked {n}, got {len(sample)}"


def test_a_small_stratum_is_not_silently_dropped():
    sample = select(CORPUS, subset=100)
    assert composition(sample).get("open-domain", 0) > 0


def test_the_sample_comes_back_in_dataset_order():
    """Shards are cut from this list by history_id (see datasets.shard),
    and ingestion cost depends on a history's questions staying together."""
    sample = select(CORPUS, subset=200)
    positions = [CORPUS.index(q) for q in sample]
    assert positions == sorted(positions)


def test_asking_for_more_than_the_corpus_returns_the_corpus():
    assert len(select(CORPUS, subset=10_000)) == len(CORPUS)
    assert len(select(CORPUS, subset=None)) == len(CORPUS)


def test_the_type_filter_still_applies_before_sampling():
    sample = select(CORPUS, subset=50, types=["temporal", "multi-hop"])
    assert set(composition(sample)) == {"temporal", "multi-hop"}
    assert len(sample) == 50


def test_a_single_type_corpus_is_not_a_special_case():
    only = [replace(q) for q in CORPUS if q.qtype == "temporal"]
    sample = select(only, subset=30)
    assert len(sample) == 30
    assert set(composition(sample)) == {"temporal"}
