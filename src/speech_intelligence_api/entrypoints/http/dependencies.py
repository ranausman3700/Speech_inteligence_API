"""FastAPI request dependencies."""

from typing import Annotated

from fastapi import Request, Response, Security
from fastapi.security import APIKeyHeader

from speech_intelligence_api.adapters.control_center import ControlCenterClient
from speech_intelligence_api.domain.errors import AuthenticationError
from speech_intelligence_api.entrypoints.http.rate_limiting import (
    acquire_rate_limit,
    caller_identity,
    set_rate_limit_headers,
)
from speech_intelligence_api.entrypoints.http.security import ApiKeyAuthenticator, ApiPrincipal
from speech_intelligence_api.ports.rate_limiting import RateLimiter

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyHeader",
    description="Deployment-issued API key.",
    auto_error=False,
)


async def authenticate_api_key(
    request: Request,
    response: Response,
    candidate: Annotated[str | None, Security(api_key_header)],
) -> ApiPrincipal:
    """Authenticate the caller without retaining the supplied credential."""

    principal = await authenticate_api_key_only(request, candidate)
    settings = request.app.state.settings
    if not settings.rate_limit_enabled:
        return principal

    upload_paths = {
        f"{settings.api_prefix}/transcriptions",
        f"{settings.api_prefix}/conversations",
    }
    is_upload = request.method == "POST" and request.url.path in upload_paths
    scope = "upload" if is_upload else "http"
    limit = settings.rate_limit_upload_requests if is_upload else settings.rate_limit_http_requests
    limiter: RateLimiter | None = request.app.state.rate_limiter
    decision = await acquire_rate_limit(
        limiter,
        caller_identity(principal, request.client.host if request.client else None),
        scope=scope,
        limit=limit,
        window_seconds=settings.rate_limit_window_seconds,
    )
    set_rate_limit_headers(response.headers, decision)
    return principal


async def authenticate_api_key_only(
    request: Request,
    candidate: Annotated[str | None, Security(api_key_header)],
) -> ApiPrincipal:
    """Authenticate an operator scrape without consuming public API capacity.

    Checks the static, deployment-wide key list first (no network call). Only
    when that fails, and the Control Center integration is enabled, does it
    ask the dashboard whether this is a dashboard-issued, still-valid token —
    this is what lets the boss revoke a specific developer's access instantly
    without touching this deployment's own configuration.
    """

    authenticator: ApiKeyAuthenticator = request.app.state.authenticator
    try:
        return authenticator.authenticate(candidate)
    except AuthenticationError:
        control_center: ControlCenterClient | None = request.app.state.control_center_client
        if control_center is not None and candidate:
            result = await control_center.verify_token(candidate)
            if result.valid:
                return ApiPrincipal(identifier=f"control-center:{result.developer or 'token'}")
        raise
