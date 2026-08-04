import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreateKeyRequest(BaseModel):
    org_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    label: str | None = Field(default=None, max_length=128)


class KeyCreatedResponse(BaseModel):
    """The clear key is returned ONLY here, at creation. It is never stored
    (sha256 hash only) and never shown again."""

    id: uuid.UUID
    key: str
    prefix: str
    org_id: str
    project_id: str
    label: str | None
    created_at: datetime


class KeyOut(BaseModel):
    """Masked view: prefix only, never the key nor its hash."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prefix: str
    org_id: str
    project_id: str
    label: str | None
    created_at: datetime
    revoked_at: datetime | None


class KeyListResponse(BaseModel):
    keys: list[KeyOut]


class KeyRevokedResponse(BaseModel):
    id: uuid.UUID
    status: str = "revoked"
