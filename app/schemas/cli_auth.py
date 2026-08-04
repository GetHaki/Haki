"""CLI device-code auth flow (sprint 14) — see app/api/routes/cli_auth.py
for the full contract and the security rationale of each field."""

from pydantic import BaseModel, Field


class DeviceStartResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


class DevicePollRequest(BaseModel):
    device_code: str = Field(min_length=1)


class DevicePollResponse(BaseModel):
    status: str  # "pending" | "approved" | "expired"
    api_key: str | None = None
    org_id: str | None = None
    project_id: str | None = None


class DeviceApproveRequest(BaseModel):
    user_code: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)


class DeviceApproveResponse(BaseModel):
    ok: bool
