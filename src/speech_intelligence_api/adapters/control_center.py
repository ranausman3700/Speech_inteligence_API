"""Optional integration with the CodesBunny API Control Center dashboard.

The dashboard is a separate, boss-only service that issues per-developer
tokens (independent of the static `SPEECH_API_API_KEY_DIGESTS` list) and can
revoke any of them instantly. Every call here is best-effort: a slow or
unreachable Control Center must never block or fail a real transcription
request, so failures are logged and swallowed rather than raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ControlCenterVerifyResult:
    """Outcome of asking the Control Center whether a token is still allowed."""

    valid: bool
    developer: str | None = None
    reason: str | None = None


class ControlCenterClient:
    """Thin async client for the Control Center's verify/log endpoints."""

    def __init__(self, *, base_url: str, service_key: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-Service-Key": service_key},
            timeout=timeout_seconds,
        )

    async def verify_token(self, raw_token: str) -> ControlCenterVerifyResult:
        """Check a token issued by the dashboard. Never raises."""

        try:
            response = await self._client.post("/api/v1/verify/", json={"token": raw_token})
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Control Center verify call failed: %s", exc)
            return ControlCenterVerifyResult(valid=False, reason="control_center_unreachable")
        return ControlCenterVerifyResult(
            valid=bool(data.get("valid")),
            developer=data.get("developer"),
            reason=data.get("reason"),
        )

    async def log_usage(
        self,
        *,
        token: str | None,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        ip_address: str | None,
        user_agent: str | None,
    ) -> None:
        """Report a completed request for the boss's usage dashboard. Never raises."""

        payload = {
            "token": token or "",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "ip_address": ip_address,
            "user_agent": user_agent or "",
        }
        try:
            response = await self._client.post("/api/v1/log/", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Control Center log call failed: %s", exc)

    async def aclose(self) -> None:
        await self._client.aclose()
