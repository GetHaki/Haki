import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class FactOut(BaseModel):
    """Full read view of a fact, all statuses (console listing)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: str
    project_id: str
    subject_type: str
    subject_id: str
    agent_id: str | None
    predicate: str
    value: dict[str, Any]
    qualifiers: dict[str, Any]
    status: str
    confidence: float | None
    valid_from: datetime | None
    valid_to: datetime | None
    recorded_from: datetime
    recorded_to: datetime | None
    supersedes_id: uuid.UUID | None
    source_event_ids: list[uuid.UUID]
    version: int


class FactListResponse(BaseModel):
    facts: list[FactOut]


class TraceSummaryOut(BaseModel):
    """Trace list entry (no packet/decisions payload: fetch /v1/inspect/{id})."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: str
    subject_id: str
    query: str
    purpose: str | None
    token_count: int
    created_at: datetime
    duration_ms: int | None = None
    fact_count: int | None = None


class TraceListResponse(BaseModel):
    traces: list[TraceSummaryOut]
