"""Lexical (full-text) retrieval axis: query parsing and tsquery construction.

Why this module exists
-----------------------
Until 20 Aug the lexical axis was one inline call in the context query:

    ts_query = func.websearch_to_tsquery("simple", query)

(a first pass that night narrowed it to `func.websearch_to_tsquery("english",
query)` -- config fixed, AND semantics kept; superseded by this module.)

Two properties of the original call, both invisible from the call site,
made the 0.25-weighted full-text term of the hybrid score contribute
*nothing* on almost every real query:

1. `websearch_to_tsquery` joins terms with `&` (AND) -- documented,
   PostgreSQL manual 12.3. A user question is a sentence, so the AND
   requires EVERY word of the question to appear in the document. One
   missing word, no match at all.
2. The `'simple'` text search configuration applies neither stemming nor
   stopword removal (manual 12.6), so "when", "did", "to", "the" are all
   required terms too, and "Caroline" never matches "caroline's".

The OR form is not a relaxation of quality, it is the standard division of
labour in retrieval: the boolean query decides what is *eligible*, ranking
decides what is *good*. Requiring every term is a filter, and a filter is
the wrong tool for recall.

Design notes
------------
- **Two steps, one extra roundtrip.** The lexemes are read from Postgres
  (only Postgres knows the stemmer and stopword list of a configuration),
  then the tsquery text is assembled in Python and bound back as a
  parameter. The alternative -- one clever nested SQL expression -- saves
  a fraction of a millisecond on a code path that already spends tens of
  milliseconds embedding the query, and costs the two things that matter
  more here: the tsquery becomes *observable* (available for the context
  trace) and *unit-testable* without a database.

- **`::tsquery` cast, never `to_tsquery`.** `to_tsquery` re-runs the
  dictionary over its input, including over quoted lexemes: verified on
  this project's own Postgres 16,
  `to_tsquery('english', $$'carolin' | 'running'$$)` returns
  `'carolin' | 'run'` -- the already-stemmed `'running'` lexeme gets
  stemmed a second time. Feeding already-stemmed lexemes back through the
  stemmer is stemming twice; a plain cast parses the tsquery syntax
  without any dictionary pass.

- **Escaping is ours.** `quote_literal()` was the obvious candidate and is
  wrong: on a lexeme containing a backslash it emits PostgreSQL's escape
  form `E'back\\slash'`, and that leading `E` is not valid tsquery syntax
  -- verified: casting it directly raises, and passing its literal
  characters through as inert parameter text instead silently drops the
  backslash from the parsed lexeme (`::tsquery` treats a backslash as its
  own escape character). tsquery quoting doubles both a single quote and
  a backslash inside `'...'` (verified, same session).

- **NULL is the empty query.** A question made only of stopwords yields no
  lexeme; this module then returns SQL NULL cast to tsquery. `vector @@
  NULL` is NULL (so no row is selected) and `ts_rank_cd(vector, NULL)` is
  NULL (so `coalesce(..., 0.0)` scores it 0). No special case at the call
  sites, no error path, no query rewriting.

- **`config` is cast to `regconfig` explicitly.** `to_tsvector(config,
  query)` with `config` bound as an ordinary text parameter fails on this
  project's Postgres with "function to_tsvector(text, text) does not
  exist" -- Postgres only implicitly casts an UNTYPED string *literal* to
  `regconfig`, never a typed bind parameter (verified via a PREPARE'd
  statement). The explicit `cast(..., REGCONFIG)` below is required, not
  decorative.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import ColumnElement, cast, literal, null, select
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.config import settings

# Inside a quoted tsquery lexeme, `'` and `\` are each doubled. Verified on
# this project's Postgres 16: `$$'o''brien'$$::tsquery` and
# `$$'back\\slash'$$::tsquery` both parse to the intended single-character
# content; `quote_literal()`'s `E'back\\slash'` form does not survive a
# round trip through `::tsquery` intact.
_TSQUERY_ESCAPES = str.maketrans({"'": "''", "\\": "\\\\"})

# (table, source column, index name) -- the two GENERATED tsvector columns
# whose configuration must always match `text_search_config()` below. Used
# by `rebuild_statements` (migrations and `scripts/set_fts_config.py` share
# this instead of each hand-writing the DDL).
FTS_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("facts", "search_text", "ix_facts_search_vector"),
    ("events", "index_text", "ix_events_search_vector"),
)


def text_search_config() -> str:
    """The text search configuration used for BOTH indexing and querying.

    It is a single value on purpose: a tsquery built with one configuration
    and matched against a tsvector built with another silently returns
    nothing (english stems `caroline` to `carolin`, simple does not). The
    stored `search_vector` columns are GENERATED, so the configuration is
    baked into the schema by whichever migration created them -- see
    `app.db.verify_fts_config`, which refuses to start on a mismatch
    rather than let the lexical axis quietly go dark again.
    """
    return settings.fts_config


def lexemes_to_or_tsquery(lexemes: Sequence[str]) -> str | None:
    """Assemble already-normalized lexemes into an OR tsquery string.

    Returns None when there is nothing to search for (empty query, or a
    query made entirely of stopwords) -- see the module docstring on why
    that becomes SQL NULL rather than an exception.
    """
    if not lexemes:
        return None
    return " | ".join(f"'{lexeme.translate(_TSQUERY_ESCAPES)}'" for lexeme in lexemes)


async def build_query_tsquery(
    session: AsyncSession, query: str, *, config: str | None = None
) -> tuple[ColumnElement[TSQUERY], str | None]:
    """Turn a user question into an OR tsquery bound for the search columns.

    Returns `(sql_expression, debug_text)`. `debug_text` is None when the
    query holds no searchable lexeme; it is meant for diagnostics (e.g. a
    future context-trace field), not for control flow -- the SQL
    expression is already NULL-safe.
    """
    config = config or text_search_config()
    lexemes: list[str] | None = await session.scalar(
        select(func.tsvector_to_array(func.to_tsvector(cast(config, REGCONFIG), query)))
    )
    tsquery_text = lexemes_to_or_tsquery(lexemes or [])
    if tsquery_text is None:
        return cast(null(), TSQUERY), None
    return cast(literal(tsquery_text), TSQUERY), tsquery_text


def rebuild_statements(config: str) -> list[str]:
    """The DDL that rebuilds both GENERATED tsvector columns under `config`.

    Returned as data rather than executed, so that a migration and
    `scripts/set_fts_config.py` run the exact same statements over two
    different execution contexts (Alembic's op.execute vs. a plain async
    engine) without duplicating the DDL -- one source of truth.

    Postgres does not allow ALTER on a GENERATED column's expression, so
    this DROPs and re-ADDs it -- a full table rewrite under an ACCESS
    EXCLUSIVE lock. Fine at beta volumes; do it out of band (build the new
    column and index CONCURRENTLY under a temporary name, then swap) on a
    large table.
    """
    if not config.isidentifier():
        # Interpolated into DDL below: a text search configuration name is
        # an identifier, never arbitrary text.
        raise ValueError(f"invalid text search configuration name: {config!r}")
    statements: list[str] = []
    for table, source, index in FTS_COLUMNS:
        statements += [
            f"DROP INDEX IF EXISTS {index}",
            f"ALTER TABLE {table} DROP COLUMN IF EXISTS search_vector",
            (
                f"ALTER TABLE {table} ADD COLUMN search_vector tsvector "
                f"GENERATED ALWAYS AS (to_tsvector('{config}', coalesce({source}, ''))) STORED"
            ),
            f"CREATE INDEX {index} ON {table} USING gin (search_vector)",
        ]
    return statements
