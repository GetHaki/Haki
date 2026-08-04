import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EventOut(BaseModel):
    id: uuid.UUID
    org_id: str
    project_id: str
    subject_type: str
    subject_id: str
    actor_type: str | None
    actor_id: str | None
    agent_id: str | None
    thread_id: str | None
    run_id: str | None
    kind: str
    occurred_at: datetime
    recorded_at: datetime
    payload: dict[str, Any]
    source: dict[str, Any] | None
    classification: list[str]
    retention_policy: str | None
    hash: str
    idempotency_key: str

    model_config = {"from_attributes": True}


class TimelineResponse(BaseModel):
    events: list[EventOut]
