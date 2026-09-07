"""Redis-backed distributed sliding-window rate limiting."""

from __future__ import annotations

import hashlib
import math
import re
import secrets

from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from speech_intelligence_api.application.readiness import ReadinessCheck
from speech_intelligence_api.domain.errors import DependencyUnavailableError
from speech_intelligence_api.ports.rate_limiting import RateLimitDecision

_MAX_TRANSACTION_RETRIES = 64
_SCOPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class RedisSlidingWindowRateLimiter:
    """Enforce exact rolling-window limits with bounded optimistic transactions."""

    def __init__(self, client: Redis, *, key_prefix: str) -> None:
        self._client = client
        self._key_prefix = key_prefix

    async def acquire(
        self,
        identity: str,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        if not identity:
            raise ValueError("rate-limit identity cannot be empty")
        if not _SCOPE_PATTERN.fullmatch(scope):
            raise ValueError("rate-limit scope is invalid")
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate-limit values must be positive")

        identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        key = f"{self._key_prefix}:rate:{scope}:{identity_digest}"
        window_ms = window_seconds * 1000

        for _ in range(_MAX_TRANSACTION_RETRIES):
            try:
                async with self._client.pipeline(transaction=True) as pipeline:
                    await pipeline.watch(key)
                    now_ms = await self._redis_time_ms()
                    window_start_ms = now_ms - window_ms
                    active_count = int(await pipeline.zcount(key, f"({window_start_ms}", "+inf"))
                    oldest = await pipeline.zrangebyscore(
                        key,
                        f"({window_start_ms}",
                        "+inf",
                        start=0,
                        num=1,
                        withscores=True,
                    )
                    oldest_ms = int(oldest[0][1]) if oldest else now_ms
                    reset_after = max(
                        1,
                        math.ceil((oldest_ms + window_ms - now_ms) / 1000),
                    )

                    pipeline.multi()  # type: ignore[no-untyped-call]
                    pipeline.zremrangebyscore(key, "-inf", window_start_ms)
                    if active_count < limit:
                        member = f"{now_ms}:{secrets.token_hex(8)}"
                        pipeline.zadd(key, {member: now_ms})
                    pipeline.pexpire(key, window_ms + 1000)
                    await pipeline.execute()

                if active_count < limit:
                    return RateLimitDecision(
                        allowed=True,
                        limit=limit,
                        remaining=limit - active_count - 1,
                        retry_after_seconds=0,
                        reset_after_seconds=reset_after,
                    )
                return RateLimitDecision(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    retry_after_seconds=reset_after,
                    reset_after_seconds=reset_after,
                )
            except WatchError:
                continue
            except RedisError:
                raise DependencyUnavailableError from None
        raise DependencyUnavailableError

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except RedisError:
            raise DependencyUnavailableError from None

    async def _redis_time_ms(self) -> int:
        seconds, microseconds = await self._client.time()
        return int(seconds) * 1000 + int(microseconds) // 1000


class RedisRateLimitReadinessCheck(ReadinessCheck):
    """Expose rate-limit Redis readiness without connection details."""

    def __init__(self, limiter: RedisSlidingWindowRateLimiter) -> None:
        self._limiter = limiter

    @property
    def name(self) -> str:
        return "redis"

    async def check(self) -> None:
        await self._limiter.ping()
