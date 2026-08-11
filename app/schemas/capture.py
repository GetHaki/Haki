import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.event import ORIGIN_TRUST_LEVELS
from app.schemas.subjects import SubjectAliasIn


class EventIn(BaseModel):
    """Source event, contract B.1. subject_id is validated by the Ledger
    (typed `missing_scope` error), not by Pydantic."""

    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    subject_type: str = Field(default="user", max_length=64)
    subject_id: str | None = Field(default=None, max_length=128)
    # Identity resolution (M4): the client backend may address the subject
    # by a channel alias instead of a subject_id; the capture route resolves
    # it BEFORE the policy scope check. Mutually exclusive with subject_id.
    subject_alias: SubjectAliasIn | None = None

    actor_type: str | None = Field(default=None, max_length=64)
    actor_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    thread_id: str | None = Field(default=None, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
    # Origin trust (M8): what the caller can honestly assert about where
    # this content came from. Omitted -> derived server-side from
    # actor_type (agent/tool/system -> semi_trusted, else trusted). Only
    # the authenticated backend ever sets this — no model-facing surface
    # exposes it as a parameter.
    origin_trust: str | None = Field(
        default=None, pattern="^(" + "|".join(ORIGIN_TRUST_LEVELS) + ")$"
    )

    kind: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    payload: dict[str, Any]
    source: dict[str, Any] | None = None
    classification: list[str] = Field(default_factory=list)
    retention_policy: str | None = Field(default=None, max_length=128)
    idempotency_key: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def subject_or_alias(self) -> "EventIn":
        if self.subject_id is not None and self.subject_alias is not None:
            raise ValueError("subject_id and subject_alias are mutually exclusive")
        return self


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
