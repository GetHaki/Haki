import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ProvisionOrgRequest(BaseModel):
    """Provisioning is driven by an already-verified external identity
    (Clerk): the console backend is the only trusted caller (see
    HAKI_CONSOLE_SERVICE_KEY), so this never accepts org_id/project_id —
    those are always server-generated here, unlike the free-string
    self-hosted/curl bootstrap in POST /v1/keys."""

    owner_ref: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=128)


class ProvisionOrgResponse(BaseModel):
    """Always returns a fresh, usable clear key. A repeat call for an
    owner_ref that already has an Organization does NOT recreate it (that
    org_id/project_id stay stable) — it mints one more key on the existing
    project instead, since the original key's clear value cannot be
    recovered (only its hash is stored, same contract as POST /v1/keys).
    `org_created` tells the caller which case happened."""

    org_id: uuid.UUID
    project_id: str
    api_key: str
    org_created: bool
    created_at: datetime
