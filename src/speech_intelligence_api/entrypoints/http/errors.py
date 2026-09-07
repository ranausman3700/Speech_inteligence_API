"""Sanitized HTTP exception mapping."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from speech_intelligence_api.domain.errors import (
    CapacityExceededError,
    ErrorCode,
    RateLimitExceededError,
    ServiceError,
)
from speech_intelligence_api.entrypoints.http.schemas import ProblemDetail, ValidationIssue

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR_CODE = {
    ErrorCode.AUTHENTICATION_FAILED: HTTPStatus.UNAUTHORIZED,
    ErrorCode.PAYLOAD_TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
    ErrorCode.INVALID_AUDIO: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.AUDIO_TOO_LONG: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.LANGUAGE_UNCERTAIN: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.ASYNC_PROCESSING_REQUIRED: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.MODEL_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.INVALID_REQUEST: HTTPStatus.UNPROCESSABLE_ENTITY,
    ErrorCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    ErrorCode.CONFLICT: HTTPStatus.CONFLICT,
    ErrorCode.RATE_LIMITED: HTTPStatus.TOO_MANY_REQUESTS,
    ErrorCode.CAPACITY_EXCEEDED: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.DEPENDENCY_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
    ErrorCode.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def _trace_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def _problem_response(
    request: Request,
    *,
    status: HTTPStatus,
    code: ErrorCode,
    detail: str,
    issues: list[ValidationIssue] | None = None,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    trace_id = _trace_id(request)
    problem = ProblemDetail(
        type=f"urn:speech-intelligence:error:{code.value}",
        title=status.phrase,
        status=int(status),
        detail=detail,
        code=code.value,
        instance=f"urn:request:{trace_id}",
        trace_id=trace_id,
        issues=issues,
        details=details,
    )
    return JSONResponse(
        status_code=int(status),
        content=problem.model_dump(mode="json", exclude_none=True),
        media_type="application/problem+json",
        headers=headers,
    )


def service_error_response(request: Request, exc: ServiceError) -> JSONResponse:
    """Map one expected service failure to the public problem contract."""

    status = _STATUS_BY_ERROR_CODE[exc.code]
    headers: dict[str, str] | None = None
    if status is HTTPStatus.UNAUTHORIZED:
        headers = {"WWW-Authenticate": "ApiKey"}
    elif isinstance(exc, CapacityExceededError):
        headers = {"Retry-After": str(exc.retry_after_seconds)}
    elif isinstance(exc, RateLimitExceededError):
        headers = {
            "Retry-After": str(exc.retry_after_seconds),
            "X-RateLimit-Limit": str(exc.limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset-After": str(exc.retry_after_seconds),
        }
    return _problem_response(
        request,
        status=status,
        code=exc.code,
        detail=exc.public_message,
        details=exc.details or None,
        headers=headers,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers that never expose internal exception details."""

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return service_error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        issues = [
            ValidationIssue(
                location=".".join(str(part) for part in error["loc"]),
                error_type=str(error["type"]),
            )
            for error in exc.errors()
        ]
        return _problem_response(
            request,
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            code=ErrorCode.INVALID_REQUEST,
            detail="The request did not satisfy the API contract.",
            issues=issues,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        try:
            status = HTTPStatus(exc.status_code)
        except ValueError:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        code = ErrorCode.NOT_FOUND if status is HTTPStatus.NOT_FOUND else ErrorCode.INVALID_REQUEST
        return _problem_response(
            request,
            status=status,
            code=code,
            detail=status.description,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled request failure",
            extra={"exception_class": type(exc).__name__},
            exc_info=True,
        )
        return _problem_response(
            request,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            code=ErrorCode.INTERNAL_ERROR,
            detail="The service could not complete the request.",
        )
