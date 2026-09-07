"""Multipart asynchronous conversation API contract tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from fastapi.testclient import TestClient

from speech_intelligence_api.application.conversations import (
    ConversationSubmissionService,
    ConversationUploadCommand,
    QueuedConversationOutcome,
)
from speech_intelligence_api.application.transcriptions import BatchTranscriptionService
from speech_intelligence_api.domain.enums import JobKind, JobStatus, LanguageCode
from speech_intelligence_api.domain.models import JobRecord
from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import TEST_API_KEY, make_settings


class UnusedTranscriptionService:
    async def execute(self, command: object) -> object:
        raise AssertionError("batch transcription must not be called")


class ConversationService:
    def __init__(self) -> None:
        self.command: ConversationUploadCommand | None = None
        self.audio = b""

    async def execute(
        self,
        command: ConversationUploadCommand,
    ) -> QueuedConversationOutcome:
        self.command = command
        self.audio = b"".join([chunk async for chunk in command.chunks])
        now = datetime.now(tz=UTC)
        return QueuedConversationOutcome(
            JobRecord(
                job_id="job_" + "c" * 32,
                kind=JobKind.CONVERSATION,
                status=JobStatus.QUEUED,
                created_at=now,
                expires_at=now + timedelta(minutes=20),
            ),
            replayed=False,
        )


def _client(service: ConversationService | None) -> TestClient:
    settings = make_settings().model_copy(
        update={"async_jobs_enabled": True, "diarization_enabled": True}
    )
    return TestClient(
        create_app(
            settings,
            transcription_service=cast(
                BatchTranscriptionService,
                UnusedTranscriptionService(),
            ),
            conversation_service=cast(ConversationSubmissionService, service),
        )
    )


def test_conversation_submission_returns_dedicated_202_contract() -> None:
    service = ConversationService()

    with _client(service) as client:
        response = client.post(
            "/v1/conversations",
            headers={
                "X-API-Key": TEST_API_KEY,
                "Idempotency-Key": "conversation-retry-0001",
            },
            files={"file": ("meeting.wav", b"private-audio", "audio/wav")},
            data={
                "language_mode": "explicit",
                "language": "en",
                "expected_speakers": "3",
                "vocabulary": ["Codex", "FastAPI"],
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["task"] == "diarize"
    assert payload["kind"] == "conversation"
    assert payload["links"]["result"].endswith("/result")
    assert service.audio == b"private-audio"
    assert service.command is not None
    assert service.command.expected_speakers == 3
    assert service.command.language.language is LanguageCode.ENGLISH
    assert service.command.idempotency_key == "conversation-retry-0001"


def test_conversation_validates_options_before_calling_service() -> None:
    service = ConversationService()

    with _client(service) as client:
        response = client.post(
            "/v1/conversations",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("meeting.wav", b"private-audio", "audio/wav")},
            data={"language_mode": "explicit", "language": "zh"},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert "private-audio" not in response.text
    assert service.command is None


def test_conversation_requires_auth_and_reports_disabled_dependency() -> None:
    service = ConversationService()

    with _client(service) as client:
        unauthenticated = client.post(
            "/v1/conversations",
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
        )
    with _client(None) as client:
        unavailable = client.post(
            "/v1/conversations",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
        )

    assert unauthenticated.status_code == 401
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "dependency_unavailable"


def test_capabilities_and_openapi_advertise_conversations() -> None:
    service = ConversationService()

    with _client(service) as client:
        capability = client.get(
            "/v1/capabilities",
            headers={"X-API-Key": TEST_API_KEY},
        ).json()["conversations"]
        operation = client.get("/openapi.json").json()["paths"]["/v1/conversations"]["post"]

    assert capability == {
        "endpoint": "/v1/conversations",
        "asynchronous_only": True,
        "expected_speakers_minimum": 2,
        "expected_speakers_maximum": 20,
        "word_timestamps": True,
        "overlap_detection": True,
    }
    assert operation["operationId"] == "create_conversation"
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert set(operation["responses"]) >= {"202", "413", "415", "422", "503"}
