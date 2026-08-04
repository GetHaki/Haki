import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    """Source event, contract B.1. subject_id is validated by the Ledger
    (typed `missing_scope` error), not by Pydantic."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    subject_type: str = Field(default="user", max_length=64)
    subject_id: str | None = Field(default=None, max_length=128)

    actor_type: str | None = Field(default=None, max_length=64)
    actor_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    thread_id: str | None = Field(default=None, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)

    kind: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    payload: dict[str, Any]
    source: dict[str, Any] | None = None
    classification: list[str] = Field(default_factory=list)
    retention_policy: str | None = Field(default=None, max_length=128)
    idempotency_key: str | None = Field(default=None, max_length=256)


class CaptureRequest(BaseModel):
    events: list[EventIn] = Field(min_length=1)
    idempotency_key: str | None = Field(default=None, max_length=256)


class CapturedEvent(BaseModel):
    id: uuid.UUID
    deduplicated: bool


class CaptureResponse(BaseModel):
    status: str = "accepted"
    events: list[CapturedEvent]
    # None when the whole batch was deduplicated: nothing new to consolidate.
    consolidation_job_id: uuid.UUID | None
    policy: str
