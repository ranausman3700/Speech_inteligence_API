"""End-to-end application tests for queued transcription and job controls."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import fakeredis.aioredis
import pytest
from redis.asyncio import Redis

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.native_text import NativeTranscriptTextProcessor
from speech_intelligence_api.adapters.pyav_audio import PyAvAudioPreprocessor
from speech_intelligence_api.adapters.redis_jobs import RedisJobStore
from speech_intelligence_api.application.jobs import (
    JobManagementService,
    TranscriptionJobWorker,
)
from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionService,
    QueuedTranscriptionOutcome,
    UploadCommand,
)
from speech_intelligence_api.domain.enums import (
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
    ProcessingMode,
)
from speech_intelligence_api.domain.errors import ErrorCode, JobNotReadyError, ServiceError
from speech_intelligence_api.domain.jobs import TranscriptionJobResult
from speech_intelligence_api.domain.models import (
    LanguageSelection,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.jobs import JobDispatcher
from tests.audio_fixtures import make_wav_bytes


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


class RecordingDispatcher:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str, int]] = []
        self.revoked: list[str] = []

    async def enqueue(self, job_id: str, *, queue: str, priority: int) -> str:
        self.enqueued.append((job_id, queue, priority))
        return f"task-{len(self.enqueued)}"

    async def revoke(self, task_id: str) -> None:
        self.revoked.append(task_id)


class DeterministicRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.calls += 1
        segment = TranscriptSegment(
            text="hello",
            start_seconds=0,
            end_seconds=0.1,
            language=LanguageCode.ENGLISH,
        )
        return TranscriptionResult(
            language=LanguageCode.ENGLISH,
            language_confidence_estimate=0.99,
            text="hello",
            segments=(segment,),
        )


def _redis_store() -> tuple[RedisJobStore, Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisJobStore(client, key_prefix="async-test"), client


def _entries(path: Path) -> list[Path]:
    return list(path.iterdir())


def _command(
    audio: bytes,
    *,
    idempotency_key: str | None = None,
) -> UploadCommand:
    return UploadCommand(
        filename="voice.wav",
        declared_media_type="audio/wav",
        chunks=_chunks(audio),
        language=LanguageSelection(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ENGLISH,
        ),
        processing_mode=ProcessingMode.ASYNC,
        idempotency_key=idempotency_key,
    )


@pytest.mark.asyncio
async def test_async_submission_is_idempotent_and_worker_persists_result(
    tmp_path: Path,
) -> None:
    blob_store = LocalEphemeralBlobStore(tmp_path)
    job_store, redis_client = _redis_store()
    dispatcher = RecordingDispatcher()
    recognizer = DeterministicRecognizer()
    service = BatchTranscriptionService(
        store=blob_store,
        preprocessor=PyAvAudioPreprocessor(blob_store),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        max_upload_bytes=1024 * 1024,
        max_audio_duration_seconds=60,
        sync_max_audio_duration_seconds=1,
        privacy_ttl_seconds=300,
        job_store=job_store,
        job_dispatcher=cast(JobDispatcher, dispatcher),
        max_pending_jobs=1000,
        long_audio_queue_threshold_seconds=30,
    )
    audio = make_wav_bytes(duration_seconds=0.1)
    key = "retry-safe-key-0001"

    first = await service.execute(_command(audio, idempotency_key=key))
    replay = await service.execute(_command(audio, idempotency_key=key))

    assert isinstance(first, QueuedTranscriptionOutcome)
    assert isinstance(replay, QueuedTranscriptionOutcome)
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.job.job_id == first.job.job_id
    assert dispatcher.enqueued == [(first.job.job_id, "transcription.short", 7)]
    assert len(_entries(tmp_path)) == 1

    worker = TranscriptionJobWorker(
        job_store=job_store,
        blob_store=blob_store,
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
    )
    await worker.execute(first.job.job_id)
    completed = await job_store.get(first.job.job_id)
    result = await job_store.get_result(first.job.job_id)

    assert completed is not None
    assert completed.status is JobStatus.SUCCEEDED
    assert isinstance(result, TranscriptionJobResult)
    assert result.result.text == "hello"
    assert recognizer.calls == 1
    assert _entries(tmp_path) == []
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_management_cancels_queued_job_and_running_delete_conflicts(
    tmp_path: Path,
) -> None:
    blob_store = LocalEphemeralBlobStore(tmp_path)
    job_store, redis_client = _redis_store()
    dispatcher = RecordingDispatcher()
    recognizer = DeterministicRecognizer()
    service = BatchTranscriptionService(
        store=blob_store,
        preprocessor=PyAvAudioPreprocessor(blob_store),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        max_upload_bytes=1024 * 1024,
        max_audio_duration_seconds=60,
        sync_max_audio_duration_seconds=1,
        privacy_ttl_seconds=300,
        job_store=job_store,
        job_dispatcher=cast(JobDispatcher, dispatcher),
    )
    first = await service.execute(_command(make_wav_bytes()))
    assert isinstance(first, QueuedTranscriptionOutcome)
    management = JobManagementService(
        job_store=job_store,
        dispatcher=cast(JobDispatcher, dispatcher),
        blob_store=blob_store,
    )

    cancelled = await management.cancel(first.job.job_id)

    assert cancelled.status is JobStatus.CANCELLED
    assert dispatcher.revoked == ["task-1"]
    assert _entries(tmp_path) == []
    with pytest.raises(JobNotReadyError):
        await management.result(first.job.job_id)
    await management.delete(first.job.job_id)

    second = await service.execute(_command(make_wav_bytes()))
    assert isinstance(second, QueuedTranscriptionOutcome)
    await job_store.transition(second.job.job_id, JobStatus.RUNNING, progress_percent=10)
    with pytest.raises(ServiceError) as captured:
        await management.delete(second.job.job_id)
    assert captured.value.code is ErrorCode.CONFLICT
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_worker_honors_cancellation_and_sanitizes_exhausted_failure(
    tmp_path: Path,
) -> None:
    blob_store = LocalEphemeralBlobStore(tmp_path)
    job_store, redis_client = _redis_store()
    dispatcher = RecordingDispatcher()
    recognizer = DeterministicRecognizer()
    service = BatchTranscriptionService(
        store=blob_store,
        preprocessor=PyAvAudioPreprocessor(blob_store),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        max_upload_bytes=1024 * 1024,
        max_audio_duration_seconds=60,
        sync_max_audio_duration_seconds=1,
        privacy_ttl_seconds=300,
        job_store=job_store,
        job_dispatcher=cast(JobDispatcher, dispatcher),
    )
    queued = await service.execute(_command(make_wav_bytes()))
    assert isinstance(queued, QueuedTranscriptionOutcome)
    await job_store.request_cancellation(queued.job.job_id)
    worker = TranscriptionJobWorker(
        job_store=job_store,
        blob_store=blob_store,
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
    )

    await worker.execute(queued.job.job_id)
    cancelled = await job_store.get(queued.job.job_id)

    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert recognizer.calls == 0

    failed_job = await service.execute(_command(make_wav_bytes()))
    assert isinstance(failed_job, QueuedTranscriptionOutcome)
    await job_store.transition(failed_job.job.job_id, JobStatus.RUNNING, progress_percent=10)
    await worker.fail(failed_job.job.job_id, ErrorCode.DEPENDENCY_UNAVAILABLE.value)
    failed = await job_store.get(failed_job.job.job_id)

    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.failure_code == "dependency_unavailable"
    assert _entries(tmp_path) == []
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_redelivered_terminal_task_cleans_crash_orphaned_audio(tmp_path: Path) -> None:
    blob_store = LocalEphemeralBlobStore(tmp_path)
    job_store, redis_client = _redis_store()
    dispatcher = RecordingDispatcher()
    recognizer = DeterministicRecognizer()
    service = BatchTranscriptionService(
        store=blob_store,
        preprocessor=PyAvAudioPreprocessor(blob_store),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        max_upload_bytes=1024 * 1024,
        max_audio_duration_seconds=60,
        sync_max_audio_duration_seconds=1,
        privacy_ttl_seconds=300,
        job_store=job_store,
        job_dispatcher=cast(JobDispatcher, dispatcher),
    )
    queued = await service.execute(_command(make_wav_bytes()))
    assert isinstance(queued, QueuedTranscriptionOutcome)
    await job_store.transition(
        queued.job.job_id,
        JobStatus.FAILED,
        failure_code="worker_lost",
    )
    worker = TranscriptionJobWorker(
        job_store=job_store,
        blob_store=blob_store,
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
    )

    await worker.execute(queued.job.job_id)

    assert _entries(tmp_path) == []
    assert recognizer.calls == 0
    await redis_client.aclose()
