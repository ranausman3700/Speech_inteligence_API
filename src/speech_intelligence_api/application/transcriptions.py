"""Secure synchronous batch-transcription use case."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from speech_intelligence_api.application.audio_uploads import (
    DETECTED_MEDIA_BY_EXTENSION,
    validate_upload_identity,
)
from speech_intelligence_api.domain.enums import (
    JobKind,
    JobQueue,
    JobStatus,
    ProcessingMode,
)
from speech_intelligence_api.domain.errors import (
    AsyncProcessingRequiredError,
    AudioTooLongError,
    DependencyUnavailableError,
    PayloadTooLargeError,
    ServiceError,
    UnsupportedMediaTypeError,
)
from speech_intelligence_api.domain.jobs import TranscriptionJobPayload
from speech_intelligence_api.domain.models import (
    BlobReference,
    JobRecord,
    LanguageSelection,
    PreparedAudio,
    TranscriptionRequest,
    TranscriptionResult,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.audio import AudioPreprocessor
from speech_intelligence_api.ports.jobs import JobDispatcher, JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore
from speech_intelligence_api.ports.text import TranscriptTextProcessor


@dataclass(frozen=True, slots=True)
class UploadCommand:
    """Transport-neutral uploaded audio and transcription options."""

    filename: str
    declared_media_type: str
    chunks: AsyncIterable[bytes]
    language: LanguageSelection
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    vocabulary: tuple[str, ...] = ()
    word_timestamps: bool = True
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if len(self.vocabulary) > 100:
            raise ValueError("custom vocabulary cannot exceed 100 entries")
        if any(not item.strip() or len(item) > 100 for item in self.vocabulary):
            raise ValueError("custom vocabulary entries must contain 1 to 100 characters")
        if self.idempotency_key is not None and not 16 <= len(self.idempotency_key) <= 128:
            raise ValueError("idempotency key must contain 16 to 128 characters")


@dataclass(frozen=True, slots=True)
class BatchTranscriptionOutcome:
    """Completed direct transcription plus authoritative audio duration."""

    result: TranscriptionResult
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class QueuedTranscriptionOutcome:
    """Accepted asynchronous transcription job."""

    job: JobRecord
    replayed: bool


TranscriptionOutcome = BatchTranscriptionOutcome | QueuedTranscriptionOutcome


class BatchTranscriptionService:
    """Orchestrate temporary upload, normalization, ASR, and guaranteed cleanup."""

    def __init__(
        self,
        *,
        store: EphemeralBlobStore,
        preprocessor: AudioPreprocessor,
        recognizer: SpeechRecognizer,
        text_processor: TranscriptTextProcessor,
        max_upload_bytes: int,
        max_audio_duration_seconds: float,
        sync_max_audio_duration_seconds: float,
        privacy_ttl_seconds: int,
        job_store: JobStore | None = None,
        job_dispatcher: JobDispatcher | None = None,
        max_pending_jobs: int = 1000,
        long_audio_queue_threshold_seconds: float = 600,
    ) -> None:
        self._store = store
        self._preprocessor = preprocessor
        self._recognizer = recognizer
        self._text_processor = text_processor
        self._max_upload_bytes = max_upload_bytes
        self._max_audio_duration_seconds = max_audio_duration_seconds
        self._sync_max_audio_duration_seconds = sync_max_audio_duration_seconds
        self._privacy_ttl_seconds = privacy_ttl_seconds
        self._job_store = job_store
        self._job_dispatcher = job_dispatcher
        self._max_pending_jobs = max_pending_jobs
        self._long_audio_queue_threshold_seconds = long_audio_queue_threshold_seconds

    async def execute(self, command: UploadCommand) -> TranscriptionOutcome:
        """Transcribe short audio inline or atomically enqueue asynchronous work."""

        extension, media_type = validate_upload_identity(
            command.filename,
            command.declared_media_type,
        )
        if command.processing_mode is ProcessingMode.ASYNC and not self.jobs_enabled:
            raise AsyncProcessingRequiredError(self._sync_max_audio_duration_seconds)

        expires_at = datetime.now(tz=UTC) + timedelta(seconds=self._privacy_ttl_seconds)
        raw: BlobReference | None = None
        normalized: BlobReference | None = None
        try:
            raw = await self._store.put(
                self._bounded_chunks(command.chunks),
                media_type=media_type,
                expires_at=expires_at,
            )
            probe = await self._preprocessor.probe(raw)
            if probe.detected_media_type not in DETECTED_MEDIA_BY_EXTENSION[extension]:
                raise UnsupportedMediaTypeError
            if (
                probe.duration_seconds is not None
                and probe.duration_seconds > self._max_audio_duration_seconds
            ):
                raise AudioTooLongError(self._max_audio_duration_seconds)

            normalized, duration_seconds = await self._preprocessor.normalize(
                raw,
                expires_at=expires_at,
                max_duration_seconds=self._max_audio_duration_seconds,
            )
            prepared = PreparedAudio(
                raw=raw,
                normalized=normalized,
                probe=probe,
                duration_seconds=duration_seconds,
            )
            should_queue = command.processing_mode is ProcessingMode.ASYNC or (
                command.processing_mode is ProcessingMode.AUTO
                and prepared.duration_seconds > self._sync_max_audio_duration_seconds
            )
            if (
                command.processing_mode is ProcessingMode.SYNC
                and prepared.duration_seconds > self._sync_max_audio_duration_seconds
            ):
                raise AsyncProcessingRequiredError(self._sync_max_audio_duration_seconds)
            if should_queue:
                if not self.jobs_enabled:
                    raise AsyncProcessingRequiredError(self._sync_max_audio_duration_seconds)
                outcome = await self._enqueue(prepared, command)
                if not outcome.replayed:
                    normalized = None
                return outcome

            result = await self._recognizer.transcribe(
                TranscriptionRequest(
                    audio=prepared.normalized,
                    language=command.language,
                    vocabulary=command.vocabulary,
                    word_timestamps=command.word_timestamps,
                )
            )
            return BatchTranscriptionOutcome(
                result=self._text_processor.process(result),
                duration_seconds=prepared.duration_seconds,
            )
        finally:
            if normalized is not None:
                await self._store.delete(normalized)
            if raw is not None:
                await self._store.delete(raw)

    @property
    def jobs_enabled(self) -> bool:
        """Return whether queue dependencies were composed."""

        return self._job_store is not None and self._job_dispatcher is not None

    async def _enqueue(
        self,
        prepared: PreparedAudio,
        command: UploadCommand,
    ) -> QueuedTranscriptionOutcome:
        if self._job_store is None or self._job_dispatcher is None:
            raise AsyncProcessingRequiredError(self._sync_max_audio_duration_seconds)
        queue = (
            JobQueue.LONG_AUDIO
            if prepared.duration_seconds >= self._long_audio_queue_threshold_seconds
            else JobQueue.SHORT_TRANSCRIPTION
        )
        now = datetime.now(tz=UTC)
        job = JobRecord(
            job_id=f"job_{secrets.token_hex(16)}",
            kind=JobKind.TRANSCRIPTION,
            status=JobStatus.QUEUED,
            created_at=now,
            expires_at=prepared.normalized.expires_at,
        )
        payload = TranscriptionJobPayload(
            request=TranscriptionRequest(
                audio=prepared.normalized,
                language=command.language,
                vocabulary=command.vocabulary,
                word_timestamps=command.word_timestamps,
            ),
            duration_seconds=prepared.duration_seconds,
            queue=queue,
        )
        idempotency_digest: str | None = None
        request_fingerprint: str | None = None
        if command.idempotency_key is not None:
            idempotency_digest = hashlib.sha256(command.idempotency_key.encode("utf-8")).hexdigest()
            request_fingerprint = await self._request_fingerprint(payload)
        try:
            reservation = await self._job_store.reserve(
                job,
                payload,
                max_pending_jobs=self._max_pending_jobs,
                idempotency_digest=idempotency_digest,
                request_fingerprint=request_fingerprint,
            )
            if not reservation.created:
                return QueuedTranscriptionOutcome(job=reservation.record, replayed=True)
            try:
                task_id = await self._job_dispatcher.enqueue(
                    job.job_id,
                    queue=queue.value,
                    priority=7 if queue is JobQueue.SHORT_TRANSCRIPTION else 3,
                )
                attached = await self._job_store.attach_task(job.job_id, task_id)
                if attached is None:
                    await self._job_dispatcher.revoke(task_id)
                    raise DependencyUnavailableError
            except ServiceError:
                raise
            except Exception:
                await self._job_store.transition(
                    job.job_id,
                    JobStatus.FAILED,
                    failure_code="dependency_unavailable",
                )
                raise DependencyUnavailableError from None
            return QueuedTranscriptionOutcome(job=attached, replayed=False)
        except ServiceError:
            raise
        except Exception:
            raise DependencyUnavailableError from None

    async def _request_fingerprint(self, payload: TranscriptionJobPayload) -> str:
        audio_digest = hashlib.sha256()
        async for chunk in self._store.read_chunks(payload.request.audio):
            audio_digest.update(chunk)
        options = {
            "audio_sha256": audio_digest.hexdigest(),
            "duration_seconds": round(payload.duration_seconds, 6),
            "language_mode": payload.request.language.mode.value,
            "language": (
                payload.request.language.language.value
                if payload.request.language.language is not None
                else None
            ),
            "chinese_script": (
                payload.request.language.chinese_script.value
                if payload.request.language.chinese_script is not None
                else None
            ),
            "vocabulary": payload.request.vocabulary,
            "word_timestamps": payload.request.word_timestamps,
        }
        encoded = json.dumps(
            options,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _bounded_chunks(self, chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        total_bytes = 0
        async for chunk in chunks:
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > self._max_upload_bytes:
                raise PayloadTooLargeError(self._max_upload_bytes)
            yield chunk
