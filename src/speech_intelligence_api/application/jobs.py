"""Asynchronous job query, cancellation, deletion, and worker use cases."""

from __future__ import annotations

from datetime import UTC, datetime

from speech_intelligence_api.domain.enums import JobStatus
from speech_intelligence_api.domain.errors import (
    DependencyUnavailableError,
    ErrorCode,
    JobNotFoundError,
    JobNotReadyError,
    ServiceError,
)
from speech_intelligence_api.domain.jobs import (
    JobResult,
    TranscriptionJobPayload,
    TranscriptionJobResult,
)
from speech_intelligence_api.domain.models import BlobReference, JobRecord
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.jobs import JobDispatcher, JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore
from speech_intelligence_api.ports.text import TranscriptTextProcessor


class JobManagementService:
    """Expose safe controls over ephemeral asynchronous jobs."""

    def __init__(
        self,
        *,
        job_store: JobStore,
        dispatcher: JobDispatcher,
        blob_store: EphemeralBlobStore,
    ) -> None:
        self._job_store = job_store
        self._dispatcher = dispatcher
        self._blob_store = blob_store

    async def get(self, job_id: str) -> JobRecord:
        try:
            record = await self._job_store.get(job_id)
        except Exception:
            raise DependencyUnavailableError from None
        if record is None:
            raise JobNotFoundError
        return record

    async def result(self, job_id: str) -> JobResult:
        record = await self.get(job_id)
        if record.status is not JobStatus.SUCCEEDED:
            raise JobNotReadyError
        try:
            result = await self._job_store.get_result(job_id)
        except Exception:
            raise DependencyUnavailableError from None
        if result is None:
            raise JobNotReadyError
        return result

    async def cancel(self, job_id: str) -> JobRecord:
        record = await self.get(job_id)
        if record.status.terminal:
            return record
        try:
            updated = await self._job_store.request_cancellation(job_id)
            if updated is None:
                raise JobNotFoundError
            if record.task_id is not None:
                await self._dispatcher.revoke(record.task_id)
            if record.status is JobStatus.QUEUED:
                await self._delete_payload_audio(job_id)
                cancelled = await self._job_store.transition(job_id, JobStatus.CANCELLED)
                if cancelled is None:
                    raise JobNotFoundError
                return cancelled
            return updated
        except ServiceError:
            raise
        except Exception:
            raise DependencyUnavailableError from None

    async def delete(self, job_id: str) -> None:
        record = await self.get(job_id)
        if record.status in {JobStatus.RUNNING, JobStatus.CANCELLING}:
            raise ServiceError(
                ErrorCode.CONFLICT,
                "Cancel the running job and wait for cancellation before deleting it.",
            )
        if record.status is JobStatus.QUEUED:
            await self.cancel(job_id)
        try:
            await self._delete_payload_audio(job_id)
            deleted = await self._job_store.delete(job_id)
        except Exception:
            raise DependencyUnavailableError from None
        if not deleted:
            raise JobNotFoundError

    async def _delete_payload_audio(self, job_id: str) -> None:
        payload = await self._job_store.get_payload(job_id)
        if payload is not None:
            await self._blob_store.delete(payload.request.audio)


class TranscriptionJobWorker:
    """Execute one queued transcription with cooperative cancellation."""

    def __init__(
        self,
        *,
        job_store: JobStore,
        blob_store: EphemeralBlobStore,
        recognizer: SpeechRecognizer,
        text_processor: TranscriptTextProcessor,
    ) -> None:
        self._job_store = job_store
        self._blob_store = blob_store
        self._recognizer = recognizer
        self._text_processor = text_processor

    async def execute(self, job_id: str) -> None:
        record = await self._job_store.get(job_id)
        payload = await self._job_store.get_payload(job_id)
        if record is None or not isinstance(payload, TranscriptionJobPayload):
            return
        if record.expires_at <= datetime.now(tz=UTC):
            await self._blob_store.delete(payload.request.audio)
            await self._job_store.delete(job_id)
            return
        if record.status.terminal:
            await self._blob_store.delete(payload.request.audio)
            return
        if record.cancellation_requested or record.status is JobStatus.CANCELLING:
            await self._cancel(job_id, payload.request.audio)
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

        result = self._text_processor.process(await self._recognizer.transcribe(payload.request))
        latest = await self._job_store.get(job_id)
        if latest is None or latest.cancellation_requested or latest.status is JobStatus.CANCELLING:
            await self._cancel(job_id, payload.request.audio)
            return
        saved = await self._job_store.save_result(
            job_id,
            TranscriptionJobResult(
                result=result,
                duration_seconds=payload.duration_seconds,
            ),
        )
        if saved is not None and saved.status is JobStatus.SUCCEEDED:
            await self._blob_store.delete(payload.request.audio)
        elif saved is not None and saved.status is JobStatus.CANCELLING:
            await self._cancel(job_id, payload.request.audio)
        elif saved is None:
            await self._blob_store.delete(payload.request.audio)

    async def fail(self, job_id: str, failure_code: str) -> None:
        """Finalize an exhausted retry without exposing exception text."""

        record = await self._job_store.get(job_id)
        payload = await self._job_store.get_payload(job_id)
        if record is None or record.status.terminal:
            return
        if record.status is JobStatus.CANCELLING:
            if isinstance(payload, TranscriptionJobPayload):
                await self._cancel(job_id, payload.request.audio)
            return
        await self._job_store.transition(
            job_id,
            JobStatus.FAILED,
            failure_code=failure_code,
        )
        if isinstance(payload, TranscriptionJobPayload):
            await self._blob_store.delete(payload.request.audio)

    async def _cancel(self, job_id: str, audio: BlobReference) -> None:
        await self._blob_store.delete(audio)
        current = await self._job_store.get(job_id)
        if current is not None and current.status is JobStatus.CANCELLING:
            await self._job_store.transition(job_id, JobStatus.CANCELLED)
