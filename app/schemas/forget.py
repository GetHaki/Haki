import uuid

from pydantic import BaseModel, Field, model_validator


class ForgetRequest(BaseModel):
    """Forget one fact or one whole subject within a project.

    Exactly one of fact_id / subject_id is required; the Ledger also
    enforces it for direct (non-HTTP) callers such as the MCP tools.
    """

    project_id: str = Field(min_length=1, max_length=128)
    subject_id: str | None = Field(default=None, min_length=1, max_length=128)
    fact_id: uuid.UUID | None = None
    mode: str = Field(pattern="^(disable|delete)$")

    @model_validator(mode="after")
    def exactly_one_target(self) -> "ForgetRequest":
        if (self.fact_id is None) == (self.subject_id is None):
            raise ValueError("exactly one of fact_id or subject_id is required")
        return self


class ForgetResponse(BaseModel):
    status: str = "ok"
    mode: str
    scope: str  # fact | subject
    forget_id: uuid.UUID
    facts_disabled: int = 0
    facts_deleted: int = 0
    conflict_sets_deleted: int = 0
    events_deleted: int = 0
    traces_deleted: int = 0
