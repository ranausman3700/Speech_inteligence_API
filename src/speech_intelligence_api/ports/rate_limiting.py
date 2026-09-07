"""Distributed request-rate limiting boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """One admission decision without exposing the caller identity."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    reset_after_seconds: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("rate limit must be positive")
        if not 0 <= self.remaining <= self.limit:
            raise ValueError("remaining requests must fall within the rate limit")
        if self.retry_after_seconds < 0 or self.reset_after_seconds < 1:
            raise ValueError("rate-limit timing must be positive")
        if self.allowed and self.retry_after_seconds:
            raise ValueError("allowed requests cannot require a retry delay")
        if not self.allowed and self.retry_after_seconds < 1:
            raise ValueError("denied requests require a retry delay")


class RateLimiter(Protocol):
    """Atomically admit requests across API processes."""

    async def acquire(
        self,
        identity: str,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        """Consume one request slot and return the public-safe decision."""

    async def ping(self) -> None:
        """Raise when the backing rate-limit state is unavailable."""
