from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import ledger, policy
from app.billing.credits import (
    REASON_CAPTURE_DEBIT,
    maybe_grant_lazy_free_credits,
    resolve_billable_organization,
    try_debit_credit,
)
from app.config import settings
from app.db import get_session
from app.errors import ApiError
from app.schemas import CapturedEvent, CaptureRequest, CaptureResponse

router = APIRouter()


@router.post("/capture", response_model=CaptureResponse, status_code=202)
async def capture(
    request: CaptureRequest, session: AsyncSession = Depends(get_session)
) -> CaptureResponse:
    # Identity resolution (M4): a client backend may address the subject by
    # a channel alias; resolve (self-registering unknown aliases — first
    # contact from a channel must never fail) BEFORE the policy scope check
    # so rule 1 sees the canonical id. Everything downstream (idempotency
    # key, event hash, consolidation scope) then uses the STABLE canonical
    # subject — the whole point of the identity layer.
    for event in request.events:
        if event.subject_alias is not None:
            alias_row, _ = await ledger.resolve_alias(
                session,
                project_id=event.project_id,
                alias_kind=event.subject_alias.kind,
                alias_value=event.subject_alias.value,
                register=True,
            )
            event.subject_id = alias_row.canonical_subject_id

    # Policy Engine (rule 1): subject scope present, BEFORE any write.
    policy.check_capture_scope(request.events)
    results = await ledger.write_events(
        session, request.events, batch_idempotency_key=request.idempotency_key
    )

    # Credits (sprint 13): 1 credit per NEWLY accepted event — a
    # deduplicated replay (same idempotency key) never re-triggers
    # consolidation, so it never costs a credit either. Self-hosted org_ids
    # (no `organizations` row) resolve to None and are never checked, never
    # billed. Nothing is committed yet (write_events only flushed): raising
    # here rolls back the whole batch, including its event rows — an
    # insufficient-credit rejection never leaves a half-written batch or an
    # orphaned CreditTransaction behind.
    for event, dedup in results:
        if dedup:
            continue
        org = await resolve_billable_organization(session, event.org_id)
        if org is None:
            continue
        await maybe_grant_lazy_free_credits(session, org)
        debited = await try_debit_credit(
            session, org, reason=REASON_CAPTURE_DEBIT, reference=str(event.id)
        )
        if not debited:
            raise ApiError(
                type="insufficient_credits",
                message=(
                    f"organization {org.id} has {org.credit_balance} credit(s) "
                    "remaining, which is not enough to accept this capture"
                ),
                field="events",
                status_code=402,
            )

    new_event_ids = [event.id for event, dedup in results if not dedup]
    job = None
    if new_event_ids:
        job = await ledger.create_consolidation_job(
            session,
            project_id=request.events[0].project_id,
            event_ids=new_event_ids,
        )
    await session.commit()
    return CaptureResponse(
        events=[CapturedEvent(id=event.id, deduplicated=dedup) for event, dedup in results],
        consolidation_job_id=job.id if job else None,
        policy=settings.default_policy,
    )
