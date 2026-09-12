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

    # The "org_<uuid>" string actually stored as ApiKey.org_id — NOT the
    # bare Organization.id UUID. A caller (the console, then POST /v1/keys)
    # compares this against caller.org_id verbatim to scope key creation;
    # returning the bare UUID here silently broke every "create a second
    # key" attempt with a 403 forbidden_scope (found live, not guessed).
    org_id: str
    project_id: str
    api_key: str
    org_created: bool
    created_at: datetime


class OrgSettingsResponse(BaseModel):
    org_id: str
    name: str
    retention_days: int | None


class UpdateOrgSettingsRequest(BaseModel):
    """PATCH semantics: a field left unset keeps its current value — not
    the same as passing it explicitly as null, which for retention_days
    means "keep everything forever"."""

    owner_ref: str | None = Field(default=None, min_length=1, max_length=256)
    name: str | None = Field(default=None, min_length=1, max_length=128)
    retention_days: int | None = Field(default=None, ge=1, le=3650)
    clear_retention: bool = False  # explicit "forever" — distinct from "not sent"
