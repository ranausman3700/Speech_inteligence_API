"""Asynchronous multi-speaker conversation submission and worker use cases."""

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
from speech_intelligence_api.application.conversation_alignment import ConversationAssembler
from speech_intelligence_api.domain.enums import JobKind, JobQueue, JobStatus
from speech_intelligence_api.domain.errors import (
    AudioTooLongError,
    DependencyUnavailableError,
    PayloadTooLargeError,
    ServiceError,
    UnsupportedMediaTypeError,
)
from speech_intelligence_api.domain.jobs import (
    ConversationJobPayload,
    ConversationJobResult,
)
from speech_intelligence_api.domain.models import (
    BlobReference,
    JobRecord,
    LanguageSelection,
    TranscriptionRequest,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.audio import AudioPreprocessor
from speech_intelligence_api.ports.diarization import SpeakerDiarizer
from speech_intelligence_api.ports.jobs import JobDispatcher, JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore
from speech_intelligence_api.ports.text import TranscriptTextProcessor


@dataclass(frozen=True, slots=True)
class ConversationUploadCommand:
    """Transport-neutral private conversation upload and recognition options."""

    filename: str
    declared_media_type: str
    chunks: AsyncIterable[bytes]
    language: LanguageSelection
    expected_speakers: int | None = None
    vocabulary: tuple[str, ...] = ()
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if self.expected_speakers is not None and not 2 <= self.expected_speakers <= 100:
            raise ValueError("expected speakers must be between 2 and 100")
        if len(self.vocabulary) > 100:
            raise ValueError("custom vocabulary cannot exceed 100 entries")
        if any(not item.strip() or len(item) > 100 for item in self.vocabulary):
            raise ValueError("custom vocabulary entries must contain 1 to 100 characters")
        if self.idempotency_key is not None and not 16 <= len(self.idempotency_key) <= 128:
            raise ValueError("idempotency key must contain 16 to 128 characters")


@dataclass(frozen=True, slots=True)
class QueuedConversationOutcome:
    """Accepted diarization job and idempotent replay status."""

    job: JobRecord
    replayed: bool


class ConversationSubmissionService:
    """Normalize an upload and atomically enqueue dedicated diarization work."""

    def __init__(
        self,
        *,
        store: EphemeralBlobStore,
        preprocessor: AudioPreprocessor,
        job_store: JobStore,
        job_dispatcher: JobDispatcher,
        max_upload_bytes: int,
        max_audio_duration_seconds: float,
        privacy_ttl_seconds: int,
        max_pending_jobs: int,
        max_expected_speakers: int,
        speaker_confidence_threshold: float,
    ) -> None:
        self._store = store
        self._preprocessor = preprocessor
        self._job_store = job_store
        self._job_dispatcher = job_dispatcher
        self._max_upload_bytes = max_upload_bytes
        self._max_audio_duration_seconds = max_audio_duration_seconds
        self._privacy_ttl_seconds = privacy_ttl_seconds
        self._max_pending_jobs = max_pending_jobs
        self._max_expected_speakers = max_expected_speakers
        self._speaker_confidence_threshold = speaker_confidence_threshold

    async def execute(self, command: ConversationUploadCommand) -> QueuedConversationOutcome:
        """Always return quickly with a background job and retain no raw upload."""

        if (
            command.expected_speakers is not None
            and command.expected_speakers > self._max_expected_speakers
        ):
            raise ValueError("expected speakers exceed the configured limit")
        extension, media_type = validate_upload_identity(
            command.filename,
            command.declared_media_type,
        )
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
            payload = ConversationJobPayload(
                request=TranscriptionRequest(
                    audio=normalized,
                    language=command.language,
                    vocabulary=command.vocabulary,
                    word_timestamps=True,
                ),
                duration_seconds=duration_seconds,
                expected_speakers=command.expected_speakers,
                speaker_confidence_threshold=self._speaker_confidence_threshold,
            )
            outcome = await self._enqueue(payload, command.idempotency_key)
            if not outcome.replayed:
                normalized = None
            return outcome
        finally:
            if normalized is not None:
                await self._store.delete(normalized)
            if raw is not None:
                await self._store.delete(raw)

    async def _enqueue(
        self,
        payload: ConversationJobPayload,
        idempotency_key: str | None,
    ) -> QueuedConversationOutcome:
        now = datetime.now(tz=UTC)
        job = JobRecord(
            job_id=f"job_{secrets.token_hex(16)}",
            kind=JobKind.CONVERSATION,
            status=JobStatus.QUEUED,
            created_at=now,
            expires_at=payload.request.audio.expires_at,
        )
        idempotency_digest = None
        request_fingerprint = None
        if idempotency_key is not None:
            idempotency_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
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
                return QueuedConversationOutcome(reservation.record, replayed=True)
            try:
                task_id = await self._job_dispatcher.enqueue(
                    job.job_id,
                    queue=JobQueue.DIARIZATION.value,
                    priority=4,
                )
                attached = await self._job_store.attach_task(job.job_id, task_id)
                if attached is None:
                    await self._job_dispatcher.revoke(task_id)
                    raise DependencyUnavailableError
            except Exception:
                await self._job_store.transition(
                    job.job_id,
                    JobStatus.FAILED,
                    failure_code="dependency_unavailable",
                )
                raise DependencyUnavailableError from None
            return QueuedConversationOutcome(attached, replayed=False)
        except ServiceError:
            raise
        except Exception:
            raise DependencyUnavailableError from None

    async def _request_fingerprint(self, payload: ConversationJobPayload) -> str:
        audio_digest = hashlib.sha256()
        async for chunk in self._store.read_chunks(payload.request.audio):
            audio_digest.update(chunk)
        language = payload.request.language
        options = {
            "audio_sha256": audio_digest.hexdigest(),
            "duration_seconds": round(payload.duration_seconds, 6),
            "language_mode": language.mode.value,
            "language": language.language.value if language.language is not None else None,
            "chinese_script": (
                language.chinese_script.value if language.chinese_script is not None else None
            ),
            "vocabulary": payload.request.vocabulary,
            "expected_speakers": payload.expected_speakers,
        }
        return hashlib.sha256(
            json.dumps(
                options,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    async def _bounded_chunks(self, chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        total_bytes = 0
        async for chunk in chunks:
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > self._max_upload_bytes:
                raise PayloadTooLargeError(self._max_upload_bytes)
            yield chunk


class ConversationJobWorker:
    """Run ASR and diarization sequentially with cooperative cancellation points."""

    def __init__(
        self,
        *,
        job_store: JobStore,
        blob_store: EphemeralBlobStore,
        recognizer: SpeechRecognizer,
        text_processor: TranscriptTextProcessor,
        diarizer: SpeakerDiarizer,
        assembler: ConversationAssembler,
    ) -> None:
        self._job_store = job_store
        self._blob_store = blob_store
        self._recognizer = recognizer
        self._text_processor = text_processor
        self._diarizer = diarizer
        self._assembler = assembler

    async def execute(self, job_id: str) -> None:
        record = await self._job_store.get(job_id)
        payload = await self._job_store.get_payload(job_id)
        if (
            record is None
            or not isinstance(payload, ConversationJobPayload)
            or record.kind is not JobKind.CONVERSATION
        ):
            return
        if record.expires_at <= datetime.now(tz=UTC):
            await self._blob_store.delete(payload.request.audio)
            await self._job_store.delete(job_id)
            return
        if record.status.terminal:
            await self._blob_store.delete(payload.request.audio)
            return
        if await self._cancel_if_requested(record, payload):
            return
        if record.status is JobStatus.QUEUED:
            record = await self._job_store.transition(
                job_id,
                JobStatus.RUNNING,
                progress_percent=10,
            )
            if record is None:
                return
        elif record.status is not JobStatus.RUNNING:
            return

        transcript = self._text_processor.process(
            await self._recognizer.transcribe(payload.request)
        )
        if await self._job_store.update_progress(job_id, 45) is None:
            await self._blob_store.delete(payload.request.audio)
            return
        if await self._cancel_latest(job_id, payload):
            return
        turns = await self._diarizer.diarize(
            payload.request.audio,
            expected_speakers=payload.expected_speakers,
        )
        if await self._job_store.update_progress(job_id, 80) is None:
            await self._blob_store.delete(payload.request.audio)
            return
        if await self._cancel_latest(job_id, payload):
            return
        result = self._assembler.assemble(
            transcript,
            turns,
            speaker_confidence_threshold=payload.speaker_confidence_threshold,
        )
        saved = await self._job_store.save_result(
            job_id,
            ConversationJobResult(result=result, duration_seconds=payload.duration_seconds),
        )
        if saved is not None and saved.status is JobStatus.SUCCEEDED:
            await self._blob_store.delete(payload.request.audio)
        elif saved is not None and saved.status is JobStatus.CANCELLING:
            await self._cancel(job_id, payload.request.audio)
        elif saved is None:
            await self._blob_store.delete(payload.request.audio)

    async def fail(self, job_id: str, failure_code: str) -> None:
        record = await self._job_store.get(job_id)
        payload = await self._job_store.get_payload(job_id)
        if record is None or record.status.terminal:
            return
        if record.status is JobStatus.CANCELLING:
            if isinstance(payload, ConversationJobPayload):
                await self._cancel(job_id, payload.request.audio)
            return
        await self._job_store.transition(job_id, JobStatus.FAILED, failure_code=failure_code)
        if isinstance(payload, ConversationJobPayload):
            await self._blob_store.delete(payload.request.audio)

    async def _cancel_if_requested(
        self,
        record: JobRecord,
        payload: ConversationJobPayload,
    ) -> bool:
        if record.cancellation_requested or record.status is JobStatus.CANCELLING:
            await self._cancel(job_id=record.job_id, audio=payload.request.audio)
            return True
        return False

    async def _cancel_latest(self, job_id: str, payload: ConversationJobPayload) -> bool:
        latest = await self._job_store.get(job_id)
        if latest is not None and (
            latest.cancellation_requested or latest.status is JobStatus.CANCELLING
        ):
            await self._cancel(job_id, payload.request.audio)
            return True
        return False

    async def _cancel(self, job_id: str, audio: BlobReference) -> None:
        await self._blob_store.delete(audio)
        current = await self._job_store.get(job_id)
        if current is not None and current.status is JobStatus.CANCELLING:
            await self._job_store.transition(job_id, JobStatus.CANCELLED)
