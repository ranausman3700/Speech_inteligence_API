"""Application tests for queued multi-speaker conversation processing."""

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
from speech_intelligence_api.application.conversation_alignment import ConversationAssembler
from speech_intelligence_api.application.conversations import (
    ConversationJobWorker,
    ConversationSubmissionService,
    ConversationUploadCommand,
)
from speech_intelligence_api.domain.enums import (
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
)
from speech_intelligence_api.domain.errors import (
    AudioTooLongError,
    DependencyUnavailableError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from speech_intelligence_api.domain.jobs import ConversationJobPayload, ConversationJobResult
from speech_intelligence_api.domain.models import (
    DiarizationTurn,
    LanguageSelection,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.diarization import SpeakerDiarizer
from speech_intelligence_api.ports.jobs import JobDispatcher
from tests.audio_fixtures import make_wav_bytes


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


class Dispatcher:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str, int]] = []

    async def enqueue(self, job_id: str, *, queue: str, priority: int) -> str:
        self.enqueued.append((job_id, queue, priority))
        return "diarization-task"

    async def revoke(self, task_id: str) -> None:
        raise AssertionError(f"unexpected revoke: {task_id}")


class FailingDispatcher(Dispatcher):
    async def enqueue(self, job_id: str, *, queue: str, priority: int) -> str:
        raise OSError("broker details must be sanitized")


class Recognizer:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.calls += 1
        assert request.word_timestamps is True
        return TranscriptionResult(
            LanguageCode.ENGLISH,
            0.99,
            "hello there",
            (
                TranscriptSegment(
                    "hello there",
                    0,
                    1,
                    LanguageCode.ENGLISH,
                    words=(
                        TranscriptWord("hello ", 0, 0.5, 0.9),
                        TranscriptWord("there", 0.5, 1, 0.8),
                    ),
                ),
            ),
        )


class Diarizer:
    def __init__(self) -> None:
        self.expected_speakers: int | None = None

    async def diarize(
        self,
        audio: object,
        *,
        expected_speakers: int | None = None,
    ) -> tuple[DiarizationTurn, ...]:
        self.expected_speakers = expected_speakers
        return (
            DiarizationTurn("a", 0, 0.5),
            DiarizationTurn("b", 0.5, 1),
        )


def _store() -> tuple[RedisJobStore, Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisJobStore(client, key_prefix="conversation-test"), client


def _entries(path: Path) -> list[Path]:
    return list(path.iterdir())


def _service(
    tmp_path: Path,
    job_store: RedisJobStore,
    dispatcher: Dispatcher,
    *,
    max_upload_bytes: int = 1024 * 1024,
    max_audio_duration_seconds: float = 60,
) -> ConversationSubmissionService:
    blob_store = LocalEphemeralBlobStore(tmp_path)
    return ConversationSubmissionService(
        store=blob_store,
        preprocessor=PyAvAudioPreprocessor(blob_store),
        job_store=job_store,
        job_dispatcher=cast(JobDispatcher, dispatcher),
        max_upload_bytes=max_upload_bytes,
        max_audio_duration_seconds=max_audio_duration_seconds,
        privacy_ttl_seconds=300,
        max_pending_jobs=100,
        max_expected_speakers=20,
        speaker_confidence_threshold=0.6,
    )


def _command(
    audio: bytes,
    *,
    filename: str = "conversation.wav",
    media_type: str = "audio/wav",
    idempotency_key: str | None = None,
    expected_speakers: int | None = 2,
) -> ConversationUploadCommand:
    return ConversationUploadCommand(
        filename=filename,
        declared_media_type=media_type,
        chunks=_chunks(audio),
        language=LanguageSelection(
            LanguageSelectionMode.EXPLICIT,
            LanguageCode.ENGLISH,
        ),
        expected_speakers=expected_speakers,
        vocabulary=("Codex",),
        idempotency_key=idempotency_key,
    )


@pytest.mark.asyncio
async def test_submission_is_idempotent_and_worker_saves_diarized_result(
    tmp_path: Path,
) -> None:
    job_store, redis = _store()
    dispatcher = Dispatcher()
    service = _service(tmp_path, job_store, dispatcher)
    audio = make_wav_bytes(duration_seconds=0.1)

    first = await service.execute(_command(audio, idempotency_key="conversation-retry-0001"))
    replay = await service.execute(_command(audio, idempotency_key="conversation-retry-0001"))

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.job.job_id == first.job.job_id
    assert dispatcher.enqueued == [(first.job.job_id, "diarization", 4)]
    payload = await job_store.get_payload(first.job.job_id)
    assert isinstance(payload, ConversationJobPayload)
    assert payload.expected_speakers == 2
    assert len(_entries(tmp_path)) == 1

    recognizer = Recognizer()
    diarizer = Diarizer()
    worker = ConversationJobWorker(
        job_store=job_store,
        blob_store=LocalEphemeralBlobStore(tmp_path),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        diarizer=cast(SpeakerDiarizer, diarizer),
        assembler=ConversationAssembler(),
    )
    await worker.execute(first.job.job_id)
    record = await job_store.get(first.job.job_id)
    result = await job_store.get_result(first.job.job_id)

    assert record is not None and record.status is JobStatus.SUCCEEDED
    assert isinstance(result, ConversationJobResult)
    assert result.result.formatted_transcript == "Person 1: hello\nPerson 2: there"
    assert diarizer.expected_speakers == 2
    assert _entries(tmp_path) == []
    await redis.aclose()


@pytest.mark.asyncio
async def test_submission_rejects_limits_and_mismatched_detected_media(tmp_path: Path) -> None:
    job_store, redis = _store()
    dispatcher = Dispatcher()
    service = _service(tmp_path, job_store, dispatcher, max_upload_bytes=16)

    with pytest.raises(PayloadTooLargeError):
        await service.execute(_command(make_wav_bytes()))
    assert _entries(tmp_path) == []

    service = _service(tmp_path, job_store, dispatcher)
    with pytest.raises(UnsupportedMediaTypeError):
        await service.execute(
            _command(make_wav_bytes(), filename="fake.mp3", media_type="audio/mpeg")
        )
    assert _entries(tmp_path) == []

    with pytest.raises(ValueError):
        await service.execute(_command(make_wav_bytes(), expected_speakers=21))
    await redis.aclose()


@pytest.mark.asyncio
async def test_worker_honors_cancellation_and_cleans_failed_audio(tmp_path: Path) -> None:
    job_store, redis = _store()
    dispatcher = Dispatcher()
    service = _service(tmp_path, job_store, dispatcher)
    recognizer = Recognizer()
    worker = ConversationJobWorker(
        job_store=job_store,
        blob_store=LocalEphemeralBlobStore(tmp_path),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        diarizer=cast(SpeakerDiarizer, Diarizer()),
        assembler=ConversationAssembler(),
    )

    cancelled = await service.execute(_command(make_wav_bytes()))
    await job_store.request_cancellation(cancelled.job.job_id)
    await worker.execute(cancelled.job.job_id)
    cancelled_record = await job_store.get(cancelled.job.job_id)
    assert cancelled_record is not None and cancelled_record.status is JobStatus.CANCELLED
    assert recognizer.calls == 0

    failed = await service.execute(_command(make_wav_bytes()))
    await worker.fail(failed.job.job_id, "model_unavailable")
    failed_record = await job_store.get(failed.job.job_id)
    assert failed_record is not None and failed_record.status is JobStatus.FAILED
    assert failed_record.failure_code == "model_unavailable"
    assert _entries(tmp_path) == []
    await redis.aclose()


@pytest.mark.asyncio
async def test_redelivered_terminal_conversation_cleans_crash_orphaned_audio(
    tmp_path: Path,
) -> None:
    job_store, redis = _store()
    service = _service(tmp_path, job_store, Dispatcher())
    recognizer = Recognizer()
    queued = await service.execute(_command(make_wav_bytes()))
    await job_store.transition(
        queued.job.job_id,
        JobStatus.FAILED,
        failure_code="worker_lost",
    )
    worker = ConversationJobWorker(
        job_store=job_store,
        blob_store=LocalEphemeralBlobStore(tmp_path),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=NativeTranscriptTextProcessor(),
        diarizer=cast(SpeakerDiarizer, Diarizer()),
        assembler=ConversationAssembler(),
    )

    await worker.execute(queued.job.job_id)

    assert _entries(tmp_path) == []
    assert recognizer.calls == 0
    await redis.aclose()


def test_conversation_command_rejects_unbounded_public_options() -> None:
    common = {
        "filename": "conversation.wav",
        "declared_media_type": "audio/wav",
        "chunks": _chunks(b"audio"),
        "language": LanguageSelection(
            LanguageSelectionMode.EXPLICIT,
            LanguageCode.ENGLISH,
        ),
    }

    with pytest.raises(ValueError, match="expected speakers"):
        ConversationUploadCommand(**common, expected_speakers=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot exceed"):
        ConversationUploadCommand(**common, vocabulary=("term",) * 101)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1 to 100"):
        ConversationUploadCommand(**common, vocabulary=(" ",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="idempotency key"):
        ConversationUploadCommand(**common, idempotency_key="short")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_submission_rejects_long_audio_and_sanitizes_broker_failure(
    tmp_path: Path,
) -> None:
    job_store, redis = _store()
    too_short_limit = _service(
        tmp_path,
        job_store,
        Dispatcher(),
        max_audio_duration_seconds=0.05,
    )
    with pytest.raises(AudioTooLongError):
        await too_short_limit.execute(_command(make_wav_bytes(duration_seconds=0.1)))
    assert _entries(tmp_path) == []

    failing = _service(tmp_path, job_store, FailingDispatcher())
    with pytest.raises(DependencyUnavailableError):
        await failing.execute(_command(make_wav_bytes()))
    records = await redis.keys("conversation-test:job:*")
    assert records
    assert _entries(tmp_path) == []
    await redis.aclose()
