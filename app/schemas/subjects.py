import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


class SubjectAliasIn(BaseModel):
    """Channel identity reference, set by the CLIENT backend (never by the
    LLM — the MCP tools do not expose it). kind is a namespace ("telegram",
    "email", "device"...), normalized to lowercase; value is the channel's
    own identifier, kept verbatim (channel ids may be case-sensitive)."""

    kind: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=256)

    @field_validator("kind")
    @classmethod
    def normalize_kind(cls, v: str) -> str:
        return v.lower()


class ResolveSubjectRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    alias_kind: str = Field(min_length=1, max_length=64)
    alias_value: str = Field(min_length=1, max_length=256)
    # Omitted on first contact: the alias self-registers under the
    # deterministic canonical id "{kind}:{value}" (see ledger.resolve_alias).
    canonical_subject_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("alias_kind")
    @classmethod
    def normalize_kind(cls, v: str) -> str:
        return v.lower()


class ResolveSubjectResponse(BaseModel):
    project_id: str
    alias_kind: str
    alias_value: str
    canonical_subject_id: str
    created: bool
    self_registered: bool


class MergeSubjectsRequest(BaseModel):
    """Merge every memory of source into target within a project. The
    Ledger re-validates for direct (non-HTTP) callers, same as forget."""

    project_id: str = Field(min_length=1, max_length=128)
    source_subject_id: str = Field(min_length=1, max_length=128)
    target_subject_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def distinct_subjects(self) -> "MergeSubjectsRequest":
        if self.source_subject_id == self.target_subject_id:
            raise ValueError("source_subject_id and target_subject_id must differ")
        return self


class MergeSubjectsResponse(BaseModel):
    status: str = "ok"
    merge_id: uuid.UUID
    project_id: str
    source_subject_id: str
    target_subject_id: str
    events_moved: int = 0
    facts_moved: int = 0
    conflict_sets_moved: int = 0
    traces_moved: int = 0
    aliases_repointed: int = 0
