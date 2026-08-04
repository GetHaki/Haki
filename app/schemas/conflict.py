import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConflictSetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: str
    subject_id: str
    fact_ids: list[uuid.UUID]
    status: str
    reason: str | None
    created_at: datetime
    resolved_at: datetime | None


class ConflictListResponse(BaseModel):
    conflicts: list[ConflictSetOut]
    # Observability (sprint 10): open conflicts hide facts from every
    # context call until resolved manually — nothing auto-resolves them
    # (the "hide both, never guess" guarantee is deliberate, see PRD). This
    # summary is the counter-measure to silent accumulation: a monitoring
    # job can alert on count/oldest_open_seconds without parsing the list.
    open_count: int
    oldest_open_seconds: float | None = None


class ResolveConflictRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    keep_fact_id: uuid.UUID


class ResolveConflictResponse(BaseModel):
    conflict_id: uuid.UUID
    status: str  # "resolved"
    kept_fact_id: uuid.UUID
    superseded_fact_ids: list[uuid.UUID]
    resolved_at: datetime
