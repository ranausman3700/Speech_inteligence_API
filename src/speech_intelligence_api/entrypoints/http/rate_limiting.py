"""Transport helpers for privacy-safe caller admission control."""

from __future__ import annotations

from starlette.datastructures import MutableHeaders

from speech_intelligence_api.domain.errors import (
    DependencyUnavailableError,
    RateLimitExceededError,
)
from speech_intelligence_api.entrypoints.http.security import ApiPrincipal
from speech_intelligence_api.ports.rate_limiting import RateLimitDecision, RateLimiter

_AUTHENTICATION_DISABLED = "authentication-disabled"


def caller_identity(principal: ApiPrincipal, client_host: str | None) -> str:
    """Use the API-key identity, with an IP fallback only when authentication is disabled."""

    if principal.identifier != _AUTHENTICATION_DISABLED:
        return f"api-key:{principal.identifier}"
    return f"ip:{client_host or 'unknown'}"


async def acquire_rate_limit(
    limiter: RateLimiter | None,
    identity: str,
    *,
    scope: str,
    limit: int,
    window_seconds: int,
) -> RateLimitDecision:
    """Fail closed when distributed admission state is absent or exhausted."""

    if limiter is None:
        raise DependencyUnavailableError
    decision = await limiter.acquire(
        identity,
        scope=scope,
        limit=limit,
        window_seconds=window_seconds,
    )
    if not decision.allowed:
        raise RateLimitExceededError(
            limit=limit,
            window_seconds=window_seconds,
            retry_after_seconds=decision.retry_after_seconds,
        )
    return decision


def set_rate_limit_headers(headers: MutableHeaders, decision: RateLimitDecision) -> None:
    """Expose only non-sensitive admission metadata."""

    headers["X-RateLimit-Limit"] = str(decision.limit)
    headers["X-RateLimit-Remaining"] = str(decision.remaining)
    headers["X-RateLimit-Reset-After"] = str(decision.reset_after_seconds)
