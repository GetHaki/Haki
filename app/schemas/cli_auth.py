"""CLI device-code auth flow (sprint 14) — see app/api/routes/cli_auth.py
for the full contract and the security rationale of each field."""

from pydantic import BaseModel, Field


class DeviceStartResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    # RFC 8628 §3.2 — the same page with the code already in the query
    # string. `verification_uri` + `user_code` stay populated (a client that
    # ignores this field keeps working, and the code must remain readable
    # for a human typing it on a phone); this is for the case where a single
    # clickable link is worth more than a link plus a code to transcribe.
    verification_uri_complete: str
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
    # The console's identifier for the signed-in human doing the approving
    # (its Clerk user id). Never trusted for authorization — the service
    # key already did that — only used to count wrong code guesses per
    # person instead of per console instance (RFC 8628 §5.4).
    approver_ref: str | None = None


class DeviceApproveResponse(BaseModel):
    ok: bool
