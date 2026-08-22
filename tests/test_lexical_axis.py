"""Lexical retrieval axis: OR semantics, text search configuration, escaping.

Why these tests did not exist before, and why that mattered
-------------------------------------------------------------
The full-text term carries 0.25 of the hybrid score and, until 20 Aug,
contributed nothing on the large majority of real questions:
`websearch_to_tsquery` joins terms with AND, and the `'simple'`
configuration neither stems nor drops stopwords, so a natural-language
question matched a document only if EVERY one of its words -- "when",
"did", "the" included -- appeared in it.

Every test in the suite stayed green through all of it. Nothing asserted
what the lexical axis was supposed to *retrieve*; the axis could be
silently dead and only an end-to-end accuracy run would notice, weeks
later.

These tests pin the contract instead of the implementation: a question
that shares one meaningful term with a stored fact must retrieve that
fact.
"""

import pytest
from sqlalchemy import text

from app.config import settings
from app.context.fts import build_query_tsquery, lexemes_to_or_tsquery
from app.db import _GENERATED_FTS_CONFIG_SQL, async_session
from app.providers.fake import mock_fact
from tests.test_consolidator import capture, make_memory_event, run_worker

# FakeProvider derives embeddings from a sha256: two different texts are
# ~1.0 apart and never cluster (see app/providers/fake.py). The semantic
# axis therefore cannot retrieve anything here, which is exactly what makes
# these tests a clean probe of the LEXICAL axis alone.


def test_or_tsquery_joins_lexemes_with_or_not_and():
    assert lexemes_to_or_tsquery(["carolin", "go", "group"]) == "'carolin' | 'go' | 'group'"


def test_or_tsquery_is_none_when_nothing_is_searchable():
    """An empty query, or one made only of stopwords, has no lexeme.

    The caller turns None into SQL NULL: `vector @@ NULL` selects no row and
    `ts_rank_cd(vector, NULL)` is NULL, coalesced to 0. No exception, no
    query rewriting, no special case at the call sites.
    """
    assert lexemes_to_or_tsquery([]) is None


def test_or_tsquery_escapes_quotes_and_backslashes():
    """tsquery doubles `'` and `\\` inside a quoted lexeme.

    `quote_literal()` is the trap this replaces: on a backslash it returns
    PostgreSQL's escape-string form `E'back\\\\slash'`, and a leading `E` is
    not valid tsquery syntax.
    """
    assert lexemes_to_or_tsquery(["o'brien"]) == "'o''brien'"
    assert lexemes_to_or_tsquery(["back\\slash"]) == "'back\\\\slash'"


@pytest.mark.parametrize(
    "question",
    [
        # Only "kayak" is shared; the rest is question scaffolding.
        "When did he say he wanted to buy a kayak?",
        # Same term, inflected: 'english' stems kayaks -> kayak. Under
        # 'simple' (no stemmer) this one cannot match at all.
        "Does he own any kayaks?",
    ],
)
async def test_question_sharing_one_term_retrieves_the_fact(client, question):
    await capture(
        client,
        [make_memory_event([mock_fact("sport_equipment", {"item": "kayak"})])],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": question,
            "budget_tokens": 2000,
        },
    )
    assert response.status_code == 200
    predicates = [fact["predicate"] for fact in response.json()["packet"]["facts"]]
    assert "sport_equipment" in predicates, (
        f"the lexical axis did not retrieve the only fact sharing a term with {question!r}"
    )


async def test_query_without_searchable_terms_is_not_an_error(client):
    """A question made only of stopwords must degrade, not raise.

    `to_tsquery('')` raises a syntax error in PostgreSQL; the NULL path
    exists so that this request returns an honest packet instead of a 500.
    """
    await capture(
        client,
        [make_memory_event([mock_fact("sport_equipment", {"item": "kayak"})])],
    )
    await run_worker()

    response = await client.post(
        "/v1/context",
        json={
            "project_id": "prj_support",
            "subject_id": "usr_42",
            "query": "the and of",
            "budget_tokens": 2000,
        },
    )
    assert response.status_code == 200


async def test_build_query_tsquery_uses_the_configured_text_search_config():
    """The query side must stem exactly like the indexed side.

    'english' stems `running` to `run`; 'simple' leaves it alone. A tsquery
    built under one configuration and matched against a tsvector built
    under the other silently matches nothing -- the failure mode
    `app.db.verify_fts_config` exists to make loud.
    """
    async with async_session() as session:
        _, english = await build_query_tsquery(session, "running dogs", config="english")
        _, simple = await build_query_tsquery(session, "running dogs", config="simple")
    # tsvector lexemes come back sorted, and an OR query has no meaningful
    # order anyway — compare the sets, not the strings.
    assert set(english.split(" | ")) == {"'dog'", "'run'"}
    assert set(simple.split(" | ")) == {"'running'", "'dogs'"}


async def test_configured_text_search_config_matches_the_schema():
    """`settings.fts_config` must equal what the GENERATED columns were built with.

    This is the assertion `app.db.verify_fts_config` makes at startup; the
    test pins it for the migrated test database, so a future migration that
    changes one side without the other fails here rather than in production
    — where the only symptom would be a quietly worse ranking.
    """
    async with async_session() as session:
        rows = (await session.execute(text(_GENERATED_FTS_CONFIG_SQL))).all()
    assert rows, "no GENERATED search_vector column found — migration 0023 did not run"
    for row in rows:
        assert f"'{settings.fts_config}'::regconfig" in row.expression, (
            f"{row.table_name}.search_vector is indexed with a different text search "
            f"configuration than HAKI_FTS_CONFIG={settings.fts_config!r}"
        )


async def test_the_previous_and_semantics_would_have_matched_nothing(client):
    """Pins the regression itself, in SQL, so it cannot come back unnoticed.

    Both query forms are run against the very tsvector the fact is indexed
    with. The old form (`websearch_to_tsquery`, AND) matches nothing; the
    new one matches. Without this, a future refactor could quietly restore
    AND semantics and every other test would still pass.
    """
    await capture(
        client,
        [make_memory_event([mock_fact("sport_equipment", {"item": "kayak"})])],
    )
    await run_worker()
    question = "When did he say he wanted to buy a kayak?"

    async with async_session() as session:
        _, new_query_text = await build_query_tsquery(session, question)
        matched_new = await session.scalar(
            text("SELECT count(*) FROM facts WHERE search_vector @@ (:q)::tsquery"),
            {"q": new_query_text},
        )
        matched_old = await session.scalar(
            text(
                "SELECT count(*) FROM facts "
                "WHERE search_vector @@ websearch_to_tsquery('simple', :q)"
            ),
            {"q": question},
        )
    assert new_query_text is not None
    assert matched_old == 0, "the old AND form unexpectedly matched — rewrite this test"
    assert matched_new >= 1, "the OR form must retrieve the fact sharing one term"


async def test_startup_guard_rejects_a_configuration_mismatch(monkeypatch):
    """The guard must FIRE, not just exist.

    A drift between the queried and the indexed configuration produces no
    error and no log line on its own -- the tsquery simply stops matching.
    `verify_fts_config` converts that silent degradation into a refusal to
    serve, and this test pins that it actually raises.
    """
    from app.db import verify_fts_config

    monkeypatch.setattr(settings, "fts_config", "definitely_not_the_indexed_one")
    with pytest.raises(RuntimeError, match="text search configuration mismatch"):
        await verify_fts_config()


async def test_startup_guard_accepts_the_matching_configuration():
    from app.db import verify_fts_config

    await verify_fts_config()
