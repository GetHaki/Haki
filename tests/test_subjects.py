"""Identity resolution (M4, real database, FakeProvider):

channel alias -> canonical subject resolution (self-registration, explicit
canonical, conflict rejection), subject merge (re-scoping + receipt +
tombstone), and the fragmentation detector on /v1/context.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db import async_session
from app.models import (
    ConflictSet,
    ContextTrace,
    Event,
    Fact,
    SubjectAlias,
    SubjectMergeReceipt,
)
from app.providers.fake import mock_fact
from test_consolidator import capture, make_memory_event, run_worker

PROJECT = "prj_identity"


async def resolve(client, **payload) -> tuple[int, dict]:
    response = await client.post("/v1/subjects/resolve", json=payload)
    return response.status_code, response.json()


async def merge(client, **payload) -> tuple[int, dict]:
    response = await client.post("/v1/subjects/merge", json=payload)
    return response.status_code, response.json()


async def context(client, **payload) -> tuple[int, dict]:
    response = await client.post("/v1/context", json={"project_id": PROJECT, **payload})
    return response.status_code, response.json()


async def seed_fact(client, subject_id: str, predicate: str = "plan", value: dict | None = None) -> None:
    event = make_memory_event(
        [mock_fact(predicate, value or {"tier": "pro"}, subject_id=subject_id)],
        subject_id,
    )
    event["project_id"] = PROJECT
    await capture(client, [event])
    assert await run_worker() == 1


async def count_aliases() -> int:
    async with async_session() as session:
        rows = (await session.execute(select(SubjectAlias))).scalars().all()
        return len(rows)


async def test_resolve_self_registers_unknown_alias(client):
    status, body = await resolve(
        client, project_id=PROJECT, alias_kind="telegram", alias_value="123456"
    )
    assert status == 200
    assert body["created"] is True
    assert body["self_registered"] is True
    assert body["canonical_subject_id"] == "telegram:123456"

    async with async_session() as session:
        rows = (
            (await session.execute(select(SubjectAlias).where(SubjectAlias.project_id == PROJECT)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].alias_kind == "telegram"
    assert rows[0].alias_value == "123456"
    assert rows[0].canonical_subject_id == "telegram:123456"


async def test_resolve_with_explicit_canonical_then_idempotent_lookup(client):
    status, body = await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="42",
        canonical_subject_id="usr_42",
    )
    assert status == 200
    assert body["created"] is True
    assert body["canonical_subject_id"] == "usr_42"

    status, body = await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="42",
        canonical_subject_id="usr_42",
    )
    assert status == 200
    assert body["created"] is False
    assert body["canonical_subject_id"] == "usr_42"

    status, body = await resolve(client, project_id=PROJECT, alias_kind="telegram", alias_value="42")
    assert status == 200
    assert body["created"] is False
    assert body["canonical_subject_id"] == "usr_42"

    assert await count_aliases() == 1


async def test_resolve_conflicting_canonical_is_rejected(client):
    await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="7",
        canonical_subject_id="usr_a",
    )
    status, body = await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="7",
        canonical_subject_id="usr_b",
    )
    assert status == 409
    assert body["error"]["type"] == "alias_conflict"

    async with async_session() as session:
        row = (
            (
                await session.execute(
                    select(SubjectAlias).where(
                        SubjectAlias.project_id == PROJECT,
                        SubjectAlias.alias_value == "7",
                    )
                )
            )
            .scalars()
            .first()
        )
    assert row.canonical_subject_id == "usr_a"


async def test_resolve_kind_is_normalized_lowercase(client):
    await resolve(client, project_id=PROJECT, alias_kind="Telegram", alias_value="99")
    status, body = await resolve(client, project_id=PROJECT, alias_kind="telegram", alias_value="99")
    assert status == 200
    assert body["created"] is False
    assert await count_aliases() == 1


async def test_self_registration_too_long_requires_explicit_canonical(client):
    long_value = "v" * 200
    status, body = await resolve(client, project_id=PROJECT, alias_kind="device", alias_value=long_value)
    assert status == 422
    assert body["error"]["type"] == "alias_self_registration_too_long"

    status, body = await resolve(
        client,
        project_id=PROJECT,
        alias_kind="device",
        alias_value=long_value,
        canonical_subject_id="usr_long",
    )
    assert status == 200
    assert body["canonical_subject_id"] == "usr_long"


async def test_alias_uniqueness_enforced_by_database(client):
    async with async_session() as session:
        session.add(
            SubjectAlias(
                project_id=PROJECT,
                alias_kind="email",
                alias_value="a@example.com",
                canonical_subject_id="usr_1",
            )
        )
        await session.commit()

    async with async_session() as session:
        session.add(
            SubjectAlias(
                project_id=PROJECT,
                alias_kind="email",
                alias_value="a@example.com",
                canonical_subject_id="usr_2",
            )
        )
        try:
            await session.commit()
            raised = False
        except IntegrityError:
            raised = True
            await session.rollback()
    assert raised


async def test_capture_with_alias_stores_event_under_canonical(client):
    await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="123456",
        canonical_subject_id="usr_42",
    )
    event = {
        "org_id": "org_acme",
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_alias": {"kind": "telegram", "value": "123456"},
        "kind": "conversation.message",
        "occurred_at": "2026-08-10T10:00:00Z",
        "payload": {"role": "user", "content": "hi"},
    }
    body = await capture(client, [event])
    event_id = uuid.UUID(body["events"][0]["id"])

    async with async_session() as session:
        row = await session.get(Event, event_id)
    assert row.subject_id == "usr_42"


async def test_capture_with_unknown_alias_self_registers(client):
    event = {
        "org_id": "org_acme",
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_alias": {"kind": "device", "value": "abc-123"},
        "kind": "conversation.message",
        "occurred_at": "2026-08-10T10:00:00Z",
        "payload": {"role": "user", "content": "hi"},
    }
    body = await capture(client, [event])
    event_id = uuid.UUID(body["events"][0]["id"])

    async with async_session() as session:
        row = await session.get(Event, event_id)
        alias = (
            (
                await session.execute(
                    select(SubjectAlias).where(
                        SubjectAlias.project_id == PROJECT,
                        SubjectAlias.alias_kind == "device",
                        SubjectAlias.alias_value == "abc-123",
                    )
                )
            )
            .scalars()
            .first()
        )
    assert row.subject_id == "device:abc-123"
    assert alias is not None
    assert alias.canonical_subject_id == "device:abc-123"


async def test_capture_with_both_subject_id_and_alias_rejected(client):
    event = {
        "org_id": "org_acme",
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_id": "usr_1",
        "subject_alias": {"kind": "device", "value": "xyz"},
        "kind": "conversation.message",
        "occurred_at": "2026-08-10T10:00:00Z",
        "payload": {"role": "user", "content": "hi"},
    }
    response = await client.post(
        "/v1/capture", json={"idempotency_key": f"batch-{uuid.uuid4()}", "events": [event]}
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_payload"

    async with async_session() as session:
        rows = (
            (await session.execute(select(Event).where(Event.project_id == PROJECT)))
            .scalars()
            .all()
        )
    assert rows == []


async def test_context_with_alias_serves_canonical_memory(client):
    await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="42",
        canonical_subject_id="usr_42",
    )
    await seed_fact(client, "usr_42", predicate="plan", value={"tier": "pro"})

    status, body = await context(
        client,
        subject_alias={"kind": "telegram", "value": "42"},
        query="plan",
        purpose="test",
        budget_tokens=900,
    )
    assert status == 200
    assert len(body["packet"]["facts"]) == 1

    async with async_session() as session:
        trace = await session.get(ContextTrace, uuid.UUID(body["trace_id"]))
    assert trace.subject_id == "usr_42"


async def test_context_with_unknown_alias_fails_loudly(client):
    status, body = await context(
        client,
        subject_alias={"kind": "telegram", "value": "does-not-exist"},
        query="anything",
        purpose="test",
    )
    assert status == 404
    assert body["error"]["type"] == "alias_not_found"
    assert await count_aliases() == 0


async def test_merge_moves_all_memory_and_journals_receipt(client):
    await seed_fact(client, "usr_src", predicate="plan", value={"tier": "free"})
    await seed_fact(client, "usr_tgt", predicate="role", value={"name": "admin"})

    status, body = await merge(
        client, project_id=PROJECT, source_subject_id="usr_src", target_subject_id="usr_tgt"
    )
    assert status == 200
    assert body["facts_moved"] == 1
    assert body["events_moved"] == 1

    async with async_session() as session:
        src_facts = (
            (await session.execute(select(Fact).where(Fact.subject_id == "usr_src")))
            .scalars()
            .all()
        )
        tgt_facts = (
            (await session.execute(select(Fact).where(Fact.subject_id == "usr_tgt")))
            .scalars()
            .all()
        )
        src_events = (
            (await session.execute(select(Event).where(Event.subject_id == "usr_src")))
            .scalars()
            .all()
        )
        receipt = (
            (
                await session.execute(
                    select(SubjectMergeReceipt).where(
                        SubjectMergeReceipt.project_id == PROJECT,
                        SubjectMergeReceipt.source_subject_id == "usr_src",
                    )
                )
            )
            .scalars()
            .first()
        )
    assert src_facts == []
    assert len(tgt_facts) == 2
    assert src_events == []
    assert receipt is not None
    assert receipt.counters == {
        "events_moved": body["events_moved"],
        "facts_moved": body["facts_moved"],
        "conflict_sets_moved": body["conflict_sets_moved"],
        "traces_moved": body["traces_moved"],
        "aliases_repointed": body["aliases_repointed"],
    }
    assert receipt.created_at is not None


async def test_merge_receipt_moved_ids_are_exact(client):
    await seed_fact(client, "usr_src2", predicate="plan", value={"tier": "free"})

    async with async_session() as session:
        pre_event_ids = {
            str(e.id)
            for e in (
                await session.execute(select(Event).where(Event.subject_id == "usr_src2"))
            )
            .scalars()
            .all()
        }
        pre_fact_ids = {
            str(f.id)
            for f in (
                await session.execute(select(Fact).where(Fact.subject_id == "usr_src2"))
            )
            .scalars()
            .all()
        }

    status, body = await merge(
        client, project_id=PROJECT, source_subject_id="usr_src2", target_subject_id="usr_tgt2"
    )
    assert status == 200

    async with async_session() as session:
        receipt = (
            (
                await session.execute(
                    select(SubjectMergeReceipt).where(
                        SubjectMergeReceipt.source_subject_id == "usr_src2"
                    )
                )
            )
            .scalars()
            .first()
        )
    assert set(receipt.moved["events"]) == pre_event_ids
    assert set(receipt.moved["facts"]) == pre_fact_ids


async def test_merge_repoints_aliases_and_leaves_tombstone(client):
    await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="src1",
        canonical_subject_id="usr_src3",
    )
    status, _ = await merge(
        client, project_id=PROJECT, source_subject_id="usr_src3", target_subject_id="usr_tgt3"
    )
    assert status == 200

    async with async_session() as session:
        alias = (
            (
                await session.execute(
                    select(SubjectAlias).where(
                        SubjectAlias.project_id == PROJECT,
                        SubjectAlias.alias_kind == "telegram",
                        SubjectAlias.alias_value == "src1",
                    )
                )
            )
            .scalars()
            .first()
        )
        tombstone = (
            (
                await session.execute(
                    select(SubjectAlias).where(
                        SubjectAlias.project_id == PROJECT,
                        SubjectAlias.alias_kind == "subject",
                        SubjectAlias.alias_value == "usr_src3",
                    )
                )
            )
            .scalars()
            .first()
        )
    assert alias.canonical_subject_id == "usr_tgt3"
    assert tombstone is not None
    assert tombstone.canonical_subject_id == "usr_tgt3"

    event = {
        "org_id": "org_acme",
        "project_id": PROJECT,
        "subject_type": "user",
        "subject_alias": {"kind": "telegram", "value": "src1"},
        "kind": "conversation.message",
        "occurred_at": "2026-08-10T10:00:00Z",
        "payload": {"role": "user", "content": "hi again"},
    }
    body = await capture(client, [event])
    event_id = uuid.UUID(body["events"][0]["id"])
    async with async_session() as session:
        row = await session.get(Event, event_id)
    assert row.subject_id == "usr_tgt3"


async def test_merge_source_equals_target_rejected(client):
    status, body = await merge(
        client, project_id=PROJECT, source_subject_id="usr_same", target_subject_id="usr_same"
    )
    assert status == 422


async def test_merge_empty_source_yields_zero_counters_receipt(client):
    status, body = await merge(
        client, project_id=PROJECT, source_subject_id="usr_ghost", target_subject_id="usr_real"
    )
    assert status == 200
    assert body["events_moved"] == 0
    assert body["facts_moved"] == 0
    assert body["conflict_sets_moved"] == 0
    assert body["traces_moved"] == 0

    async with async_session() as session:
        receipt = (
            (
                await session.execute(
                    select(SubjectMergeReceipt).where(
                        SubjectMergeReceipt.source_subject_id == "usr_ghost"
                    )
                )
            )
            .scalars()
            .first()
        )
    assert receipt is not None


async def test_context_on_merged_subject_warns_fragmentation(client):
    await seed_fact(client, "usr_src4", predicate="plan", value={"tier": "free"})
    await merge(client, project_id=PROJECT, source_subject_id="usr_src4", target_subject_id="usr_tgt4")

    status, body = await context(client, subject_id="usr_src4", query="plan", purpose="test")
    assert status == 200
    assert body["packet"]["facts"] == []
    assert body["packet"]["status"] == "degraded"
    warnings = body["packet"]["warnings"]
    assert any(
        w.startswith("identity_fragmentation") and "usr_tgt4" in w for w in warnings
    )


async def test_context_on_unresolved_channel_id_warns_fragmentation(client):
    await resolve(
        client,
        project_id=PROJECT,
        alias_kind="telegram",
        alias_value="999",
        canonical_subject_id="usr_x",
    )
    await seed_fact(client, "usr_x", predicate="plan", value={"tier": "free"})

    status, body = await context(client, subject_id="999", query="plan", purpose="test")
    assert status == 200
    assert body["packet"]["status"] == "degraded"
    assert any(
        w.startswith("identity_fragmentation") and "usr_x" in w for w in body["packet"]["warnings"]
    )


async def test_true_cold_start_stays_clean(client):
    status, body = await context(
        client, subject_id="usr_never_seen", query="anything", purpose="test"
    )
    assert status == 200
    assert body["packet"]["status"] == "ok"
    assert not any(w.startswith("identity_fragmentation") for w in body["packet"]["warnings"])


async def test_rls_subject_aliases_never_disclose_other_project(client):
    from sqlalchemy import text

    async with async_session() as session:
        session.add(
            SubjectAlias(
                project_id="prj_a",
                alias_kind="telegram",
                alias_value="1",
                canonical_subject_id="usr_a",
            )
        )
        session.add(
            SubjectAlias(
                project_id="prj_b",
                alias_kind="telegram",
                alias_value="1",
                canonical_subject_id="usr_b",
            )
        )
        session.add(
            SubjectMergeReceipt(
                project_id="prj_a", source_subject_id="s", target_subject_id="t", counters={}, moved={}
            )
        )
        session.add(
            SubjectMergeReceipt(
                project_id="prj_b", source_subject_id="s", target_subject_id="t", counters={}, moved={}
            )
        )
        await session.commit()

    async with async_session() as session:
        aliases = (await session.execute(select(SubjectAlias))).scalars().all()
        receipts = (await session.execute(select(SubjectMergeReceipt))).scalars().all()
    assert {a.project_id for a in aliases} == {"prj_a", "prj_b"}
    assert {r.project_id for r in receipts} == {"prj_a", "prj_b"}

    async with async_session() as session:
        await session.execute(text("SELECT set_config('haki.project_id', 'prj_a', true)"))
        aliases = (await session.execute(select(SubjectAlias))).scalars().all()
        receipts = (await session.execute(select(SubjectMergeReceipt))).scalars().all()
    assert {a.project_id for a in aliases} == {"prj_a"}
    assert {r.project_id for r in receipts} == {"prj_a"}


async def test_rls_blocks_cross_project_alias_insert(client):
    import pytest
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    async with async_session() as session:
        await session.execute(text("SELECT set_config('haki.project_id', 'prj_a', true)"))
        session.add(
            SubjectAlias(
                project_id="prj_b",
                alias_kind="telegram",
                alias_value="intrusion",
                canonical_subject_id="usr_intrusion",
            )
        )
        with pytest.raises(ProgrammingError, match="row-level security"):
            await session.commit()
