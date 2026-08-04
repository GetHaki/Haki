import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeedbackRequest(BaseModel):
    """Quality observation on a context trace or on a fact (PRD — feedback).
    Exactly one of trace_id / fact_id is required."""

    project_id: str = Field(min_length=1, max_length=128)
    trace_id: uuid.UUID | None = None
    fact_id: uuid.UUID | None = None
    rating: Literal["useful", "irrelevant", "incorrect"]
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def exactly_one_target(self) -> "FeedbackRequest":
        if (self.trace_id is None) == (self.fact_id is None):
            raise ValueError("exactly one of trace_id or fact_id is required")
        return self


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    feedback_id: uuid.UUID
    # Resulting fact status when the feedback targeted a fact (a rating
    # `incorrect` transitions it to `disputed`), else None.
    fact_status: str | None = None
