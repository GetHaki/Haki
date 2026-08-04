from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Business error serialized as {"error": {"type", "message", "field"}}."""

    def __init__(
        self,
        type: str,
        message: str,
        field: str | None = None,
        status_code: int = 422,
    ) -> None:
        self.type = type
        self.message = message
        self.field = field
        self.status_code = status_code
        super().__init__(message)


def error_body(type: str, message: str, field: str | None = None) -> dict[str, Any]:
    return {"error": {"type": type, "message": message, "field": field}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.type, exc.message, exc.field),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(part) for part in first.get("loc", []) if part != "body"]
        field = ".".join(loc) or None
        return JSONResponse(
            status_code=422,
            content=error_body("invalid_payload", "Invalid request payload", field),
        )
