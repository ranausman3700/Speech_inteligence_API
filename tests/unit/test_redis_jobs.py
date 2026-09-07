"""Redis job-state adapter tests, including concurrent admission control."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import pytest
from redis.asyncio import Redis

from speech_intelligence_api.adapters.redis_jobs import (
    RedisClientReadinessCheck,
    RedisJobStore,
    RedisReadinessCheck,
)
from speech_intelligence_api.domain.enums import (
    JobKind,
    JobQueue,
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
)
from speech_intelligence_api.domain.errors import (
    CapacityExceededError,
    IdempotencyConflictError,
)
from speech_intelligence_api.domain.jobs import (
    ConversationJobPayload,
    ConversationJobResult,
    TranscriptionJobPayload,
    TranscriptionJobResult,
)
from speech_intelligence_api.domain.models import (
    BlobReference,
    ConversationResult,
    JobRecord,
    LanguageSelection,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)


def _store() -> tuple[RedisJobStore, Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisJobStore(client, key_prefix="test-speech"), client


def _job(index: int = 1, *, lifetime_seconds: int = 1200) -> JobRecord:
    created_at = datetime.now(tz=UTC)
    return JobRecord(
        job_id=f"job_{index:032x}",
        kind=JobKind.TRANSCRIPTION,
        status=JobStatus.QUEUED,
        created_at=created_at,
        expires_at=created_at + timedelta(seconds=lifetime_seconds),
    )


def _payload(index: int = 1) -> TranscriptionJobPayload:
    now = datetime.now(tz=UTC)
    return TranscriptionJobPayload(
        request=TranscriptionRequest(
            audio=BlobReference(
                key=f"{index:032x}.wav",
                media_type="audio/wav",
                size_bytes=100,
                created_at=now,
                expires_at=now + timedelta(minutes=20),
            ),
            language=LanguageSelection(
                mode=LanguageSelectionMode.EXPLICIT,
                language=LanguageCode.ENGLISH,
            ),
            vocabulary=("Codex",),
        ),
        duration_seconds=12.5,
        queue=JobQueue.SHORT_TRANSCRIPTION,
    )


def _result() -> TranscriptionJobResult:
    word = TranscriptWord("hello", 0.0, 0.5, 0.98)
    segment = TranscriptSegment(
        "hello",
        0.0,
        0.5,
        LanguageCode.ENGLISH,
        words=(word,),
        confidence_estimate=0.97,
    )
    return TranscriptionJobResult(
        result=TranscriptionResult(
            LanguageCode.ENGLISH,
            0.99,
            "hello",
            (segment,),
        ),
        duration_seconds=12.5,
    )


@pytest.mark.asyncio
async def test_reserve_round_trips_payload_and_does_not_extend_ttl() -> None:
    store, client = _store()
    job = _job()
    expected_payload = _payload()

    reservation = await store.reserve(job, expected_payload, max_pending_jobs=1000)
    initial_ttl = await client.pttl("test-speech:job:" + job.job_id)
    loaded = await store.get(job.job_id)
    payload = await store.get_payload(job.job_id)
    await store.attach_task(job.job_id, "task-1")
    remaining_ttl = await client.pttl("test-speech:job:" + job.job_id)

    assert reservation.created is True
    assert loaded == job
    assert payload == expected_payload
    assert 0 < remaining_ttl <= initial_ttl <= 1_200_001
    await client.aclose()


@pytest.mark.asyncio
async def test_idempotency_replays_same_job_and_rejects_changed_request() -> None:
    store, client = _store()
    first = await store.reserve(
        _job(1),
        _payload(1),
        max_pending_jobs=10,
        idempotency_digest="a" * 64,
        request_fingerprint="fingerprint-1",
    )
    replay = await store.reserve(
        _job(2),
        _payload(1),
        max_pending_jobs=10,
        idempotency_digest="a" * 64,
        request_fingerprint="fingerprint-1",
    )

    assert first.created is True
    assert replay.created is False
    assert replay.record.job_id == first.record.job_id
    with pytest.raises(IdempotencyConflictError):
        await store.reserve(
            _job(3),
            _payload(3),
            max_pending_jobs=10,
            idempotency_digest="a" * 64,
            request_fingerprint="different",
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_capacity_is_released_only_after_terminal_transition() -> None:
    store, client = _store()
    await store.reserve(_job(1), _payload(1), max_pending_jobs=1)

    with pytest.raises(CapacityExceededError):
        await store.reserve(_job(2), _payload(2), max_pending_jobs=1)

    running = await store.transition(_job(1).job_id, JobStatus.RUNNING, progress_percent=10)
    assert running is not None
    await store.transition(_job(1).job_id, JobStatus.FAILED, failure_code="model_unavailable")
    second = await store.reserve(_job(2), _payload(2), max_pending_jobs=1)

    assert second.created is True
    await client.aclose()


@pytest.mark.asyncio
async def test_result_lifecycle_cancellation_and_deletion() -> None:
    store, client = _store()
    job = _job()
    await store.reserve(job, _payload(), max_pending_jobs=10)
    await store.attach_task(job.job_id, "task-1")
    running = await store.transition(job.job_id, JobStatus.RUNNING, progress_percent=10)
    assert running is not None
    progress = await store.update_progress(job.job_id, 60)
    assert progress is not None
    succeeded = await store.save_result(job.job_id, _result())

    assert succeeded is not None
    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.progress_percent == 100
    assert await store.get_result(job.job_id) == _result()
    assert (await store.request_cancellation(job.job_id)) == succeeded
    assert await store.delete(job.job_id) is True
    assert await store.delete(job.job_id) is False
    assert await store.get(job.job_id) is None
    assert await store.get_result(job.job_id) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_result_and_idempotency_share_the_absolute_parent_expiry() -> None:
    store, client = _store()
    job = _job(lifetime_seconds=120)
    digest = "a" * 64
    await store.reserve(
        job,
        _payload(),
        max_pending_jobs=10,
        idempotency_digest=digest,
        request_fingerprint="fingerprint",
    )
    await store.transition(job.job_id, JobStatus.RUNNING, progress_percent=10)
    await store.save_result(job.job_id, _result())

    ttls = await asyncio.gather(
        client.pttl("test-speech:job:" + job.job_id),
        client.pttl("test-speech:job-result:" + job.job_id),
        client.pttl("test-speech:idempotency:" + digest),
    )

    assert all(0 < ttl <= 120_001 for ttl in ttls)
    assert max(ttls) - min(ttls) <= 10
    await client.aclose()


@pytest.mark.asyncio
async def test_orphan_result_is_fail_closed_and_physically_removed() -> None:
    store, client = _store()
    job = _job()
    await store.reserve(job, _payload(), max_pending_jobs=10)
    await store.transition(job.job_id, JobStatus.RUNNING, progress_percent=10)
    await store.save_result(job.job_id, _result())
    await client.delete("test-speech:job:" + job.job_id)

    assert await store.get_result(job.job_id) is None
    assert await client.exists("test-speech:job-result:" + job.job_id) == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_cancellation_wins_race_with_result_save() -> None:
    store, client = _store()
    job = _job()
    await store.reserve(job, _payload(), max_pending_jobs=10)
    await store.transition(job.job_id, JobStatus.RUNNING, progress_percent=10)
    cancelling = await store.request_cancellation(job.job_id)
    unchanged = await store.save_result(job.job_id, _result())

    assert cancelling is not None
    assert cancelling.status is JobStatus.CANCELLING
    assert unchanged == cancelling
    assert await store.get_result(job.job_id) is None
    await store.transition(job.job_id, JobStatus.CANCELLED)
    await client.aclose()


@pytest.mark.asyncio
async def test_accepts_one_thousand_concurrent_unique_submissions_safely() -> None:
    store, client = _store()

    reservations = await asyncio.gather(
        *(
            store.reserve(_job(index), _payload(index), max_pending_jobs=1000)
            for index in range(1, 1001)
        )
    )

    assert len(reservations) == 1000
    assert all(item.created for item in reservations)
    assert len({item.record.job_id for item in reservations}) == 1000
    assert await client.zcard("test-speech:jobs:active") == 1000
    with pytest.raises(CapacityExceededError):
        await store.reserve(_job(1001), _payload(1001), max_pending_jobs=1000)
    await client.aclose()


@pytest.mark.asyncio
async def test_readiness_and_expired_index_cleanup() -> None:
    store, client = _store()
    check = RedisReadinessCheck(store)
    await client.zadd("test-speech:jobs:active", {"expired": 1})

    await check.check()
    removed = await store.cleanup_expired_index()

    assert check.name == "redis"
    assert removed == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_isolated_broker_readiness_uses_a_bounded_role_name() -> None:
    _, client = _store()
    check = RedisClientReadinessCheck(client, name="redis_broker")

    await check.check()

    assert check.name == "redis_broker"
    with pytest.raises(ValueError, match="bounded service role"):
        RedisClientReadinessCheck(client, name="private-hostname")
    await client.aclose()


@pytest.mark.asyncio
async def test_round_trips_conversation_payload_and_speaker_result() -> None:
    store, client = _store()
    job = _job()
    job = JobRecord(
        job_id=job.job_id,
        kind=JobKind.CONVERSATION,
        status=job.status,
        created_at=job.created_at,
        expires_at=job.expires_at,
    )
    transcription_payload = _payload()
    payload = ConversationJobPayload(
        request=transcription_payload.request,
        duration_seconds=12.5,
        expected_speakers=2,
        speaker_confidence_threshold=0.6,
    )
    word = TranscriptWord("hello", 0, 0.5, 0.98)
    segment = TranscriptSegment(
        "hello",
        0,
        0.5,
        LanguageCode.ENGLISH,
        words=(word,),
        speaker="Person 1",
        speaker_confidence_estimate=0.9,
        confidence_estimate=0.98,
    )
    result = ConversationJobResult(
        ConversationResult(
            LanguageCode.ENGLISH,
            0.99,
            None,
            "hello",
            "Person 1: hello",
            (segment,),
        ),
        12.5,
    )

    await store.reserve(job, payload, max_pending_jobs=10)
    assert await store.get_payload(job.job_id) == payload
    await store.transition(job.job_id, JobStatus.RUNNING, progress_percent=10)
    await store.save_result(job.job_id, result)

    assert await store.get_result(job.job_id) == result
    await client.aclose()
