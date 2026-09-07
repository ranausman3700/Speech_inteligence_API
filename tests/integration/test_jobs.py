"""HTTP contracts for asynchronous submission and job lifecycle endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from fastapi.testclient import TestClient

from speech_intelligence_api.application.jobs import JobManagementService
from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionService,
    QueuedTranscriptionOutcome,
    UploadCommand,
)
from speech_intelligence_api.domain.enums import JobKind, JobStatus, LanguageCode
from speech_intelligence_api.domain.jobs import ConversationJobResult, TranscriptionJobResult
from speech_intelligence_api.domain.models import (
    ConversationResult,
    JobRecord,
    TranscriptionResult,
    TranscriptSegment,
)
from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import TEST_API_KEY, make_settings


def _job(status: JobStatus = JobStatus.QUEUED) -> JobRecord:
    now = datetime.now(tz=UTC)
    return JobRecord(
        job_id="job_" + "a" * 32,
        kind=JobKind.TRANSCRIPTION,
        status=status,
        created_at=now,
        expires_at=now + timedelta(minutes=20),
        progress_percent=100 if status is JobStatus.SUCCEEDED else 0,
    )


class QueuedService:
    def __init__(self) -> None:
        self.command: UploadCommand | None = None

    async def execute(self, command: UploadCommand) -> QueuedTranscriptionOutcome:
        self.command = command
        return QueuedTranscriptionOutcome(job=_job(), replayed=False)


class FakeJobService:
    def __init__(self) -> None:
        self.record = _job()
        segment = TranscriptSegment(
            "hello",
            0,
            1,
            LanguageCode.ENGLISH,
        )
        self.transcription_result = TranscriptionJobResult(
            TranscriptionResult(
                LanguageCode.ENGLISH,
                0.99,
                "hello",
                (segment,),
            ),
            1,
        )
        self.deleted: list[str] = []

    async def get(self, job_id: str) -> JobRecord:
        assert job_id == self.record.job_id
        return self.record

    async def result(
        self,
        job_id: str,
    ) -> TranscriptionJobResult | ConversationJobResult:
        assert job_id == self.record.job_id
        return self.transcription_result

    async def cancel(self, job_id: str) -> JobRecord:
        assert job_id == self.record.job_id
        return self.record.transition(JobStatus.CANCELLED)

    async def delete(self, job_id: str) -> None:
        self.deleted.append(job_id)


class ConversationJobService(FakeJobService):
    def __init__(self) -> None:
        super().__init__()
        now = datetime.now(tz=UTC)
        self.record = JobRecord(
            job_id="job_" + "b" * 32,
            kind=JobKind.CONVERSATION,
            status=JobStatus.SUCCEEDED,
            created_at=now,
            expires_at=now + timedelta(minutes=20),
            progress_percent=100,
        )
        segment = TranscriptSegment(
            "hello",
            0,
            1,
            LanguageCode.ENGLISH,
            speaker="Person 1",
            speaker_confidence_estimate=0.92,
        )
        self.conversation_result = ConversationJobResult(
            ConversationResult(
                LanguageCode.ENGLISH,
                0.99,
                None,
                "hello",
                "Person 1: hello",
                (segment,),
            ),
            1,
        )

    async def result(self, job_id: str) -> ConversationJobResult:
        assert job_id == self.record.job_id
        return self.conversation_result


def _client(
    transcription_service: QueuedService,
    job_service: FakeJobService,
) -> TestClient:
    app = create_app(
        make_settings(),
        transcription_service=cast(BatchTranscriptionService, transcription_service),
        job_service=cast(JobManagementService, job_service),
    )
    return TestClient(app)


def test_async_submission_returns_202_links_and_propagates_idempotency_key() -> None:
    transcription_service = QueuedService()
    jobs = FakeJobService()

    with _client(transcription_service, jobs) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={
                "X-API-Key": TEST_API_KEY,
                "Idempotency-Key": "retry-safe-key-0001",
            },
            files={"file": ("voice.wav", b"audio", "audio/wav")},
            data={"processing_mode": "async"},
        )

    assert response.status_code == 202
    assert response.json()["processing"] == "queued"
    assert response.json()["job_id"] == jobs.record.job_id
    assert response.json()["links"]["result"].endswith("/result")
    assert transcription_service.command is not None
    assert transcription_service.command.idempotency_key == "retry-safe-key-0001"


def test_job_status_result_cancel_and_delete_contracts() -> None:
    transcription_service = QueuedService()
    jobs = FakeJobService()
    job_id = jobs.record.job_id

    with _client(transcription_service, jobs) as client:
        status_response = client.get(
            f"/v1/jobs/{job_id}",
            headers={"X-API-Key": TEST_API_KEY},
        )
        result_response = client.get(
            f"/v1/jobs/{job_id}/result",
            headers={"X-API-Key": TEST_API_KEY},
        )
        cancel_response = client.post(
            f"/v1/jobs/{job_id}/cancel",
            headers={"X-API-Key": TEST_API_KEY},
        )
        delete_response = client.delete(
            f"/v1/jobs/{job_id}",
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"
    assert result_response.status_code == 200
    assert result_response.json()["text"] == "hello"
    assert result_response.json()["job_id"] == job_id
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"
    assert delete_response.status_code == 204
    assert jobs.deleted == [job_id]


def test_job_id_format_is_validated_before_service() -> None:
    transcription_service = QueuedService()
    jobs = FakeJobService()

    with _client(transcription_service, jobs) as client:
        response = client.get(
            "/v1/jobs/not-a-job-id",
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_conversation_job_result_returns_speaker_metadata_and_both_transcripts() -> None:
    transcription_service = QueuedService()
    jobs = ConversationJobService()

    with _client(transcription_service, jobs) as client:
        response = client.get(
            f"/v1/jobs/{jobs.record.job_id}/result",
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "diarize"
    assert payload["raw_transcript"] == "hello"
    assert payload["formatted_transcript"] == "Person 1: hello"
    assert payload["segments"][0]["speaker"] == "Person 1"
    assert payload["segments"][0]["speaker_confidence_estimate"] == 0.92
