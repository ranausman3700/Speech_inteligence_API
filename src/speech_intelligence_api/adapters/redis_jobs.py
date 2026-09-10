"""Redis-backed expiring job state with optimistic atomic updates."""

from __future__ import annotations

import json
import math
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import anyio
from redis.asyncio import Redis
from redis.exceptions import WatchError

from speech_intelligence_api.application.readiness import ReadinessCheck
from speech_intelligence_api.domain.enums import (
    ChineseScript,
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
    JobPayload,
    JobReservation,
    JobResult,
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

_MAX_TRANSACTION_RETRIES = 64
_LOCK_TTL_SECONDS = 10
_LOCK_WAIT_SECONDS = 30


class RedisJobStore:
    """Store job state in Redis with absolute TTL and CAS-style transactions."""

    def __init__(self, client: Redis, *, key_prefix: str) -> None:
        self._client = client
        self._prefix = key_prefix.rstrip(":")
        self._active_key = f"{self._prefix}:jobs:active"
        self._local_capacity_lock = anyio.Lock()

    async def reserve(
        self,
        job: JobRecord,
        payload: JobPayload,
        *,
        max_pending_jobs: int,
        idempotency_digest: str | None = None,
        request_fingerprint: str | None = None,
    ) -> JobReservation:
        """Reserve one capacity slot and idempotency key in a single transaction."""

        if (idempotency_digest is None) is not (request_fingerprint is None):
            raise ValueError("idempotency digest and fingerprint must be supplied together")
        job_key = self._job_key(job.job_id)
        idempotency_key = (
            self._idempotency_key(idempotency_digest) if idempotency_digest is not None else None
        )
        expires_at = _redis_expiry_milliseconds(job.expires_at)
        now_score = datetime.now(tz=UTC).timestamp()

        async with self._capacity_lock():
            existing = await self._existing_idempotent_job(
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
            if existing is not None:
                return JobReservation(record=existing, created=False)

            active_count = int(await self._client.zcount(self._active_key, now_score, "+inf"))
            if active_count >= max_pending_jobs:
                raise CapacityExceededError(max_pending_jobs)
            if await self._client.exists(job_key):
                raise ValueError("job ID already exists")

            mapping: dict[str, str | bytes | int | float] = {
                "record": _dump_record(job),
                "payload": _dump_payload(payload),
            }
            if idempotency_key is not None:
                mapping["idempotency_key"] = idempotency_key
            async with self._client.pipeline(transaction=True) as pipeline:
                pipeline.hset(job_key, mapping=mapping)  # type: ignore[arg-type]
                pipeline.pexpireat(job_key, expires_at)
                pipeline.zadd(
                    self._active_key,
                    {job.job_id: job.expires_at.timestamp()},
                )
                if idempotency_key is not None:
                    pipeline.hset(
                        idempotency_key,
                        mapping={
                            "job_id": job.job_id,
                            "fingerprint": request_fingerprint or "",
                        },
                    )
                    pipeline.pexpireat(idempotency_key, expires_at)
                await pipeline.execute()
            return JobReservation(record=job, created=True)

    async def get(self, job_id: str) -> JobRecord | None:
        serialized = await self._client.hget(self._job_key(job_id), "record")
        return None if serialized is None else _load_record(_as_text(serialized))

    async def get_payload(self, job_id: str) -> JobPayload | None:
        serialized = await self._client.hget(self._job_key(job_id), "payload")
        return None if serialized is None else _load_payload(_as_text(serialized))

    async def attach_task(self, job_id: str, task_id: str) -> JobRecord | None:
        return await self._mutate_record(job_id, lambda record: record.with_task_id(task_id))

    async def transition(
        self,
        job_id: str,
        target: JobStatus,
        *,
        progress_percent: int | None = None,
        failure_code: str | None = None,
    ) -> JobRecord | None:
        def mutate(record: JobRecord) -> JobRecord:
            return record.transition(
                target,
                progress_percent=progress_percent,
                failure_code=failure_code,
            )

        return await self._mutate_record(job_id, mutate, remove_from_active=target.terminal)

    async def update_progress(self, job_id: str, progress_percent: int) -> JobRecord | None:
        def mutate(record: JobRecord) -> JobRecord:
            if record.status is not JobStatus.RUNNING:
                raise ValueError("progress can only be updated for a running job")
            if not record.progress_percent <= progress_percent <= 99:
                raise ValueError("running-job progress must be monotonic and below 100")
            return replace(
                record,
                progress_percent=progress_percent,
                version=record.version + 1,
            )

        return await self._mutate_record(job_id, mutate)

    async def request_cancellation(self, job_id: str) -> JobRecord | None:
        def mutate(record: JobRecord) -> JobRecord:
            if record.status.terminal or record.status is JobStatus.CANCELLING:
                return record
            cancelling = replace(record, cancellation_requested=True)
            return cancelling.transition(JobStatus.CANCELLING)

        return await self._mutate_record(job_id, mutate)

    async def save_result(
        self,
        job_id: str,
        result: JobResult,
    ) -> JobRecord | None:
        job_key = self._job_key(job_id)
        result_key = self._result_key(job_id)
        for _ in range(_MAX_TRANSACTION_RETRIES):
            async with self._client.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(job_key, result_key)
                    serialized = await pipeline.hget(job_key, "record")
                    if serialized is None:
                        return None
                    record = _load_record(_as_text(serialized))
                    if record.cancellation_requested or record.status is JobStatus.CANCELLING:
                        return record
                    updated = record.transition(JobStatus.SUCCEEDED)
                    pipeline.multi()  # type: ignore[no-untyped-call]
                    pipeline.hset(job_key, "record", _dump_record(updated))
                    pipeline.set(result_key, _dump_result(result))
                    pipeline.pexpireat(
                        result_key,
                        _redis_expiry_milliseconds(record.expires_at),
                    )
                    pipeline.zrem(self._active_key, job_id)
                    await pipeline.execute()
                    return updated
                except WatchError:
                    continue
        raise RuntimeError("Redis result save exceeded transaction retry limit")

    async def get_result(self, job_id: str) -> JobResult | None:
        job_key = self._job_key(job_id)
        result_key = self._result_key(job_id)
        async with self._client.pipeline(transaction=True) as pipeline:
            pipeline.hget(job_key, "record")
            pipeline.get(result_key)
            serialized_record, serialized_result = await pipeline.execute()
        if serialized_record is None:
            if serialized_result is not None:
                await self._client.delete(result_key)
            return None
        record = _load_record(_as_text(serialized_record))
        if record.status is not JobStatus.SUCCEEDED:
            if serialized_result is not None:
                await self._client.delete(result_key)
            return None
        return None if serialized_result is None else _load_result(_as_text(serialized_result))

    async def delete(self, job_id: str) -> bool:
        job_key = self._job_key(job_id)
        idempotency_key = await self._client.hget(job_key, "idempotency_key")
        keys: list[str] = [job_key, self._result_key(job_id)]
        if idempotency_key is not None:
            keys.append(_as_text(idempotency_key))
        async with self._client.pipeline(transaction=True) as pipeline:
            pipeline.delete(*keys)
            pipeline.zrem(self._active_key, job_id)
            results = await pipeline.execute()
        return bool(results[0] or results[1])

    async def ping(self) -> None:
        if not await self._client.ping():
            raise ConnectionError("Redis ping did not return success")

    async def cleanup_expired_index(self, *, now: datetime | None = None) -> int:
        """Remove expired capacity-index members whose Redis records already expire."""

        cutoff = (now or datetime.now(tz=UTC)).timestamp()
        return int(await self._client.zremrangebyscore(self._active_key, "-inf", cutoff))

    async def _existing_idempotent_job(
        self,
        *,
        idempotency_key: str | None,
        request_fingerprint: str | None,
    ) -> JobRecord | None:
        if idempotency_key is None:
            return None
        values = await self._client.hgetall(idempotency_key)
        if not values:
            return None
        normalized = {_as_text(key): _as_text(value) for key, value in values.items()}
        if normalized.get("fingerprint") != request_fingerprint:
            raise IdempotencyConflictError
        existing_job_id = normalized.get("job_id")
        if existing_job_id is None:
            return None
        existing_job_key = self._job_key(existing_job_id)
        serialized = await self._client.hget(existing_job_key, "record")
        if serialized is None:
            await self._client.delete(idempotency_key)
        return None if serialized is None else _load_record(_as_text(serialized))

    @asynccontextmanager
    async def _capacity_lock(self) -> AsyncIterator[None]:
        lock_key = f"{self._prefix}:jobs:capacity-lock"
        token = secrets.token_hex(16)
        async with self._local_capacity_lock:
            with anyio.fail_after(_LOCK_WAIT_SECONDS):
                while not await self._client.set(  # noqa: ASYNC110
                    lock_key,
                    token,
                    nx=True,
                    ex=_LOCK_TTL_SECONDS,
                ):
                    await anyio.sleep(0.002)
            try:
                yield
            finally:
                await self._release_lock(lock_key, token)

    async def _release_lock(self, lock_key: str, token: str) -> None:
        for _ in range(_MAX_TRANSACTION_RETRIES):
            async with self._client.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(lock_key)
                    current = await pipeline.get(lock_key)
                    if current is None or _as_text(current) != token:
                        return
                    pipeline.multi()  # type: ignore[no-untyped-call]
                    pipeline.delete(lock_key)
                    await pipeline.execute()
                    return
                except WatchError:
                    continue

    async def _mutate_record(
        self,
        job_id: str,
        mutate: Callable[[JobRecord], JobRecord],
        *,
        remove_from_active: bool = False,
    ) -> JobRecord | None:
        job_key = self._job_key(job_id)
        for _ in range(_MAX_TRANSACTION_RETRIES):
            async with self._client.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(job_key)
                    serialized = await pipeline.hget(job_key, "record")
                    if serialized is None:
                        return None
                    updated = mutate(_load_record(_as_text(serialized)))
                    pipeline.multi()  # type: ignore[no-untyped-call]
                    pipeline.hset(job_key, "record", _dump_record(updated))
                    if remove_from_active:
                        pipeline.zrem(self._active_key, job_id)
                    await pipeline.execute()
                    return updated
                except WatchError:
                    continue
        raise RuntimeError("Redis job update exceeded transaction retry limit")

    def _job_key(self, job_id: str) -> str:
        return f"{self._prefix}:job:{job_id}"

    def _result_key(self, job_id: str) -> str:
        return f"{self._prefix}:job-result:{job_id}"

    def _idempotency_key(self, digest: str) -> str:
        return f"{self._prefix}:idempotency:{digest}"


class RedisReadinessCheck(ReadinessCheck):
    """Readiness adapter that exposes no Redis address or credentials."""

    def __init__(self, store: RedisJobStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "redis"

    async def check(self) -> None:
        await self._store.ping()


class RedisClientReadinessCheck(ReadinessCheck):
    """Check an isolated Redis role without disclosing its connection details."""

    def __init__(self, client: Redis, *, name: str) -> None:
        if name not in {"redis_broker", "redis_state"}:
            raise ValueError("Redis readiness name must identify a bounded service role")
        self._client = client
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> None:
        if not await self._client.ping():
            raise ConnectionError("Redis ping did not return success")


def _redis_expiry_milliseconds(value: datetime) -> int:
    return math.ceil(value.timestamp() * 1000)


def _as_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _dump_record(record: JobRecord) -> str:
    return _json_dump(
        {
            "job_id": record.job_id,
            "kind": record.kind.value,
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
            "progress_percent": record.progress_percent,
            "cancellation_requested": record.cancellation_requested,
            "task_id": record.task_id,
            "failure_code": record.failure_code,
            "version": record.version,
        }
    )


def _load_record(serialized: str) -> JobRecord:
    data = _json_load(serialized)
    return JobRecord(
        job_id=str(data["job_id"]),
        kind=JobKind(str(data["kind"])),
        status=JobStatus(str(data["status"])),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        expires_at=datetime.fromisoformat(str(data["expires_at"])),
        progress_percent=int(data["progress_percent"]),
        cancellation_requested=bool(data["cancellation_requested"]),
        task_id=_optional_str(data.get("task_id")),
        failure_code=_optional_str(data.get("failure_code")),
        version=int(data["version"]),
    )


def _dump_payload(payload: JobPayload) -> str:
    request = payload.request
    data: dict[str, object] = {
        "payload_kind": (
            "conversation" if isinstance(payload, ConversationJobPayload) else "transcription"
        ),
        "audio": _blob_data(request.audio),
        "language": {
            "mode": request.language.mode.value,
            "language": (
                request.language.language.value if request.language.language is not None else None
            ),
            "chinese_script": (
                request.language.chinese_script.value
                if request.language.chinese_script is not None
                else None
            ),
        },
        "vocabulary": list(request.vocabulary),
        "word_timestamps": request.word_timestamps,
        "duration_seconds": payload.duration_seconds,
        "queue": payload.queue.value,
    }
    if isinstance(payload, ConversationJobPayload):
        data.update(
            {
                "expected_speakers": payload.expected_speakers,
                "speaker_confidence_threshold": payload.speaker_confidence_threshold,
            }
        )
    return _json_dump(data)


def _load_payload(serialized: str) -> JobPayload:
    data = _json_load(serialized)
    language_data = _mapping(data["language"])
    language_value = language_data.get("language")
    script_value = language_data.get("chinese_script")
    request = TranscriptionRequest(
        audio=_load_blob(_mapping(data["audio"])),
        language=LanguageSelection(
            mode=LanguageSelectionMode(str(language_data["mode"])),
            language=LanguageCode(str(language_value)) if language_value is not None else None,
            chinese_script=ChineseScript(str(script_value)) if script_value is not None else None,
        ),
        vocabulary=tuple(str(item) for item in _list(data["vocabulary"])),
        word_timestamps=bool(data["word_timestamps"]),
    )
    if data.get("payload_kind") == "conversation":
        expected = data.get("expected_speakers")
        return ConversationJobPayload(
            request=request,
            duration_seconds=float(data["duration_seconds"]),
            expected_speakers=int(expected) if expected is not None else None,
            speaker_confidence_threshold=float(data["speaker_confidence_threshold"]),
            queue=JobQueue(str(data["queue"])),
        )
    return TranscriptionJobPayload(
        request=request,
        duration_seconds=float(data["duration_seconds"]),
        queue=JobQueue(str(data["queue"])),
    )


def _dump_result(result: JobResult) -> str:
    if isinstance(result, ConversationJobResult):
        conversation = result.result
        return _json_dump(
            {
                "result_kind": "conversation",
                "duration_seconds": result.duration_seconds,
                "language": conversation.language.value,
                "language_confidence_estimate": conversation.language_confidence_estimate,
                "chinese_script": (
                    conversation.chinese_script.value
                    if conversation.chinese_script is not None
                    else None
                ),
                "raw_transcript": conversation.raw_transcript,
                "formatted_transcript": conversation.formatted_transcript,
                "speaker_count": conversation.speaker_count,
                "segments": [_segment_data(segment) for segment in conversation.segments],
            }
        )
    transcript = result.result
    return _json_dump(
        {
            "result_kind": "transcription",
            "duration_seconds": result.duration_seconds,
            "language": transcript.language.value,
            "language_confidence_estimate": transcript.language_confidence_estimate,
            "text": transcript.text,
            "chinese_script": (
                transcript.chinese_script.value if transcript.chinese_script is not None else None
            ),
            "segments": [_segment_data(segment) for segment in transcript.segments],
        }
    )


def _load_result(serialized: str) -> JobResult:
    data = _json_load(serialized)
    script_value = data.get("chinese_script")
    segments = tuple(_load_segment(_mapping(item)) for item in _list(data["segments"]))
    if data.get("result_kind") == "conversation":
        return ConversationJobResult(
            result=ConversationResult(
                language=LanguageCode(str(data["language"])),
                language_confidence_estimate=float(data["language_confidence_estimate"]),
                chinese_script=(
                    ChineseScript(str(script_value)) if script_value is not None else None
                ),
                raw_transcript=str(data["raw_transcript"]),
                formatted_transcript=str(data["formatted_transcript"]),
                segments=segments,
                speaker_count=int(data.get("speaker_count", 0)),
            ),
            duration_seconds=float(data["duration_seconds"]),
        )
    return TranscriptionJobResult(
        result=TranscriptionResult(
            language=LanguageCode(str(data["language"])),
            language_confidence_estimate=float(data["language_confidence_estimate"]),
            text=str(data["text"]),
            segments=segments,
            chinese_script=ChineseScript(str(script_value)) if script_value is not None else None,
        ),
        duration_seconds=float(data["duration_seconds"]),
    )


def _load_segment(data: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        text=str(data["text"]),
        start_seconds=float(data["start_seconds"]),
        end_seconds=float(data["end_seconds"]),
        language=LanguageCode(str(data["language"])),
        words=tuple(_load_word(_mapping(item)) for item in _list(data["words"])),
        speaker=_optional_str(data.get("speaker")),
        speaker_uncertain=bool(data["speaker_uncertain"]),
        speaker_confidence_estimate=_optional_float(data.get("speaker_confidence_estimate")),
        overlapping_speech=bool(data["overlapping_speech"]),
        confidence_estimate=_optional_float(data.get("confidence_estimate")),
    )


def _load_word(data: dict[str, Any]) -> TranscriptWord:
    return TranscriptWord(
        text=str(data["text"]),
        start_seconds=float(data["start_seconds"]),
        end_seconds=float(data["end_seconds"]),
        confidence_estimate=_optional_float(data.get("confidence_estimate")),
    )


def _segment_data(segment: TranscriptSegment) -> dict[str, object]:
    return {
        "text": segment.text,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "language": segment.language.value,
        "speaker": segment.speaker,
        "speaker_uncertain": segment.speaker_uncertain,
        "speaker_confidence_estimate": segment.speaker_confidence_estimate,
        "overlapping_speech": segment.overlapping_speech,
        "confidence_estimate": segment.confidence_estimate,
        "words": [
            {
                "text": word.text,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
                "confidence_estimate": word.confidence_estimate,
            }
            for word in segment.words
        ],
    }


def _blob_data(reference: BlobReference) -> dict[str, object]:
    return {
        "key": reference.key,
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
        "created_at": reference.created_at.isoformat(),
        "expires_at": reference.expires_at.isoformat(),
    }


def _load_blob(data: dict[str, Any]) -> BlobReference:
    return BlobReference(
        key=str(data["key"]),
        media_type=str(data["media_type"]),
        size_bytes=int(data["size_bytes"]),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        expires_at=datetime.fromisoformat(str(data["expires_at"])),
    )


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_load(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("serialized job data must be an object")
    return loaded


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("serialized job field must be an object")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("serialized job field must be an array")
    return value


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
