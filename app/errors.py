"""One error envelope for the whole API.

    {"error": {"code": "...", "message": "...", "detail": {...}, "request_id": "..."}}

``code`` is machine-readable and stable; ``message`` is for a human. The UI
switches on ``code`` and renders ``message``. Adding a new failure mode means
adding a new code, never reusing a near-miss -- a frontend cannot branch on
prose.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """Base for every error this application raises on purpose."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.detail = detail or {}
        super().__init__(self.message)


class NotFound(ApiError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "The requested resource does not exist."


class BadRequest(ApiError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"
    message = "The request was malformed."


class Unauthorized(ApiError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "A valid X-API-Key header is required."


class Forbidden(ApiError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    message = "This API key lacks the required scope."


class RateLimited(ApiError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Rate limit exceeded for this API key."


class PayloadTooLarge(ApiError):
    status_code = 413
    code = "payload_too_large"
    message = "The uploaded file exceeds the size limit."


class UnsupportedMedia(ApiError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"
    message = "The uploaded file type is not supported."


class ScorerUnavailable(ApiError):
    """A model checkpoint this route needs is not on disk.

    Deliberately 503 and not 500: nothing is broken, a capability is absent, and
    the message names which one so the operator knows what to mount.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "scorer_unavailable"
    message = "The model required for this route is not available."


class NotImplementedYet(ApiError):
    """A route whose contract exists but whose real query has not landed yet.

    501, not 500 or 404: the route is documented and its shape is final, the
    implementation simply arrives at a later build step. Every occurrence is
    deleted as its step lands, and a test asserts none survive into the tagged
    release of a step that claims to have wired it.
    """

    status_code = 501
    code = "not_implemented"
    message = (
        "This route's contract is published but its query is not wired yet. "
        "Set DEMO_MODE=1 to develop against the fixture data."
    )


class DependencyUnavailable(ApiError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "dependency_unavailable"
    message = "A backing service is unavailable."


def _envelope(
    request: Request, *, code: str, message: str, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "detail": detail or {},
            "request_id": getattr(request.state, "request_id", None),
        }
    }


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(request, code=exc.code, message=exc.message, detail=exc.detail),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # FastAPI's own 404/405 and anything raised as a bare HTTPException.
        code = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
            409: "conflict",
            429: "rate_limited",
        }.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(request, code=code, message=str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's error list is genuinely useful to a frontend developer, so
        # it is passed through in `detail` rather than flattened to a sentence.
        return JSONResponse(
            status_code=422,
            content=_envelope(
                request,
                code="validation_error",
                message="One or more request parameters were invalid.",
                detail={"errors": _jsonable_errors(exc.errors())},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                request,
                code="internal_error",
                message="An unexpected error occurred. The request id identifies it in the logs.",
            ),
        )


def _jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pydantic v2 puts exception objects in ``ctx``; those are not JSON."""
    cleaned: list[dict[str, Any]] = []
    for err in errors:
        item = {k: v for k, v in err.items() if k != "ctx"}
        if "ctx" in err:
            item["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        item["loc"] = [str(part) for part in item.get("loc", [])]
        cleaned.append(item)
    return cleaned
