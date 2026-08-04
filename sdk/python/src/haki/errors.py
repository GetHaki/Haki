"""Typed errors raised by the Haki SDK."""

from typing import Any


class HakiError(Exception):
    """Base class for every SDK error."""


class HakiConnectionError(HakiError):
    """The API is unreachable (network error, timeout, DNS...)."""


class HakiApiError(HakiError):
    """The API returned an error payload {"error": {type, message, field}}."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str | None = None,
        field: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.field = field
        self.payload = payload or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.error_type:
            return f"[{self.status_code} {self.error_type}] {base}"
        return f"[{self.status_code}] {base}"
