"""Distributed Redis rate-limiter tests."""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from speech_intelligence_api.adapters.redis_rate_limiting import (
    RedisRateLimitReadinessCheck,
    RedisSlidingWindowRateLimiter,
)
from speech_intelligence_api.domain.errors import DependencyUnavailableError


def _limiter() -> tuple[RedisSlidingWindowRateLimiter, Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisSlidingWindowRateLimiter(client, key_prefix="test-speech"), client


@pytest.mark.asyncio
async def test_sliding_window_allows_limit_then_returns_retry_metadata() -> None:
    limiter, client = _limiter()

    first = await limiter.acquire("caller-a", scope="http", limit=2, window_seconds=60)
    second = await limiter.acquire("caller-a", scope="http", limit=2, window_seconds=60)
    denied = await limiter.acquire("caller-a", scope="http", limit=2, window_seconds=60)

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert denied.allowed is False
    assert denied.retry_after_seconds in range(1, 61)
    assert denied.reset_after_seconds == denied.retry_after_seconds
    assert all("caller-a" not in key for key in await client.keys("*"))
    await client.aclose()


@pytest.mark.asyncio
async def test_identity_and_workload_scopes_have_independent_windows() -> None:
    limiter, client = _limiter()

    await limiter.acquire("caller-a", scope="http", limit=1, window_seconds=60)
    other_identity = await limiter.acquire(
        "caller-b",
        scope="http",
        limit=1,
        window_seconds=60,
    )
    other_scope = await limiter.acquire(
        "caller-a",
        scope="live",
        limit=1,
        window_seconds=60,
    )

    assert other_identity.allowed is True
    assert other_scope.allowed is True
    await client.aclose()


@pytest.mark.asyncio
async def test_concurrent_admission_never_exceeds_limit() -> None:
    limiter, client = _limiter()

    decisions = await asyncio.gather(
        *(
            limiter.acquire("shared-caller", scope="upload", limit=10, window_seconds=60)
            for _ in range(25)
        )
    )

    assert sum(decision.allowed for decision in decisions) == 10
    assert sum(not decision.allowed for decision in decisions) == 15
    await client.aclose()


@pytest.mark.asyncio
async def test_readiness_uses_the_same_redis_dependency() -> None:
    limiter, client = _limiter()
    check = RedisRateLimitReadinessCheck(limiter)

    await check.check()

    assert check.name == "redis"
    await client.aclose()


@pytest.mark.asyncio
async def test_redis_failure_is_sanitized_as_dependency_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter, client = _limiter()

    async def fail_time() -> int:
        raise RedisError("private Redis address")

    monkeypatch.setattr(limiter, "_redis_time_ms", fail_time)

    with pytest.raises(DependencyUnavailableError):
        await limiter.acquire("caller", scope="http", limit=1, window_seconds=60)

    await client.aclose()


@pytest.mark.parametrize(
    ("identity", "scope", "limit", "window_seconds"),
    [
        ("", "http", 1, 60),
        ("caller", "bad scope", 1, 60),
        ("caller", "http", 0, 60),
        ("caller", "http", 1, 0),
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_admission_parameters(
    identity: str,
    scope: str,
    limit: int,
    window_seconds: int,
) -> None:
    limiter, client = _limiter()

    with pytest.raises(ValueError):
        await limiter.acquire(
            identity,
            scope=scope,
            limit=limit,
            window_seconds=window_seconds,
        )

    await client.aclose()
