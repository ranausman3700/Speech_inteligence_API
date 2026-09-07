"""Multipart transcription API integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.native_text import NativeTranscriptTextProcessor
from speech_intelligence_api.adapters.pyav_audio import PyAvAudioPreprocessor
from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionOutcome,
    BatchTranscriptionService,
    UploadCommand,
)
from speech_intelligence_api.domain.enums import (
    ChineseScript,
    LanguageCode,
    LanguageSelectionMode,
)
from speech_intelligence_api.domain.errors import ServiceError, UncertainLanguageError
from speech_intelligence_api.domain.models import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from speech_intelligence_api.entrypoints.http.app import create_app
from tests.audio_fixtures import make_wav_bytes
from tests.factories import TEST_API_KEY, make_settings


class FakeTranscriptionService:
    def __init__(self, failure: ServiceError | None = None) -> None:
        self.failure = failure
        self.command: UploadCommand | None = None
        self.uploaded = b""

    async def execute(self, command: UploadCommand) -> BatchTranscriptionOutcome:
        self.command = command
        self.uploaded = b"".join([chunk async for chunk in command.chunks])
        if self.failure is not None:
            raise self.failure
        word = TranscriptWord(
            text="مرحبا",
            start_seconds=0.1,
            end_seconds=0.8,
            confidence_estimate=0.97,
        )
        segment = TranscriptSegment(
            text="مرحبا",
            start_seconds=0,
            end_seconds=1,
            language=LanguageCode.ARABIC,
            words=(word,),
            confidence_estimate=0.91,
        )
        return BatchTranscriptionOutcome(
            result=TranscriptionResult(
                language=LanguageCode.ARABIC,
                language_confidence_estimate=1,
                text="مرحبا",
                segments=(segment,),
            ),
            duration_seconds=1,
        )


class PipelineRecognizer:
    def __init__(self, store: LocalEphemeralBlobStore) -> None:
        self.store = store
        self.normalized_bytes = b""

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.normalized_bytes = b"".join(
            [chunk async for chunk in self.store.read_chunks(request.audio)]
        )
        segment = TranscriptSegment(
            text="hello",
            start_seconds=0,
            end_seconds=0.1,
            language=LanguageCode.ENGLISH,
        )
        return TranscriptionResult(
            language=LanguageCode.ENGLISH,
            language_confidence_estimate=1,
            text="hello",
            segments=(segment,),
        )


def _app_with_service(
    service: FakeTranscriptionService,
    *,
    max_upload_bytes: int = 100 * 1024 * 1024,
) -> TestClient:
    settings = make_settings().model_copy(update={"max_upload_bytes": max_upload_bytes})
    app = create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, service),
    )
    return TestClient(app)


def test_transcription_accepts_multipart_and_returns_native_script() -> None:
    service = FakeTranscriptionService()
    audio = make_wav_bytes()

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY, "X-Request-ID": "transcription-123"},
            files={"file": ("voice.wav", audio, "audio/wav")},
            data={
                "language_mode": "explicit",
                "language": "ar",
                "processing_mode": "sync",
                "vocabulary": ["OpenAI", "FastAPI"],
                "word_timestamps": "true",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "transcription-123"
    assert response.json() == {
        "processing": "completed",
        "task": "transcribe",
        "request_id": "transcription-123",
        "language": "ar",
        "language_confidence_estimate": 1.0,
        "chinese_script": None,
        "duration_seconds": 1.0,
        "text": "مرحبا",
        "segments": [
            {
                "text": "مرحبا",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "language": "ar",
                "confidence_estimate": 0.91,
                "words": [
                    {
                        "text": "مرحبا",
                        "start_seconds": 0.1,
                        "end_seconds": 0.8,
                        "confidence_estimate": 0.97,
                    }
                ],
            }
        ],
    }
    assert service.uploaded == audio
    assert service.command is not None
    assert service.command.filename == "voice.wav"
    assert service.command.vocabulary == ("OpenAI", "FastAPI")
    assert service.command.language.language is LanguageCode.ARABIC


def test_transcription_requires_authentication() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            files={"file": ("voice.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"
    assert service.command is None


def test_invalid_language_combination_uses_sanitized_problem_contract() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.wav", b"private-audio", "audio/wav")},
            data={"language_mode": "explicit", "language": "zh"},
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "invalid_request"
    assert "private-audio" not in response.text
    assert service.command is None


def test_swagger_chinese_default_is_ignored_for_explicit_english() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.m4a", b"audio", "audio/mp4")},
            data={
                "language_mode": "explicit",
                "language": "en",
                "chinese_script": "simplified",
            },
        )

    assert response.status_code == 200
    assert service.command is not None
    assert service.command.language.mode is LanguageSelectionMode.EXPLICIT
    assert service.command.language.language is LanguageCode.ENGLISH
    assert service.command.language.chinese_script is None


def test_swagger_language_default_is_ignored_in_automatic_mode() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.m4a", b"audio", "audio/mp4")},
            data={
                "language_mode": "automatic",
                "language": "en",
                "chinese_script": "simplified",
            },
        )

    assert response.status_code == 200
    assert service.command is not None
    assert service.command.language.mode is LanguageSelectionMode.AUTOMATIC
    assert service.command.language.language is None
    assert service.command.language.chinese_script is ChineseScript.SIMPLIFIED


def test_uncertain_language_returns_candidates_without_internal_details() -> None:
    service = FakeTranscriptionService(UncertainLanguageError(0.75, (("en", 0.61), ("fr", 0.23))))

    with _app_with_service(service) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.wav", b"audio", "audio/wav")},
            data={"language_mode": "automatic"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "language_uncertain"
    assert payload["details"] == {
        "confidence_threshold": 0.75,
        "candidates": [
            {"language": "en", "confidence_estimate": 0.61},
            {"language": "fr", "confidence_estimate": 0.23},
        ],
    }


def test_multipart_body_limit_rejects_request_before_service() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service, max_upload_bytes=1024) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.wav", b"x" * 1_100_000, "audio/wav")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert response.json()["details"] == {"max_bytes": 1024}
    assert service.command is None


def test_transcription_openapi_documents_multipart_contract() -> None:
    service = FakeTranscriptionService()

    with _app_with_service(service) as client:
        operation = client.get("/openapi.json").json()["paths"]["/v1/transcriptions"]["post"]

    assert operation["operationId"] == "create_transcription"
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert set(operation["responses"]) >= {"200", "413", "415", "422", "503"}
    assert operation["security"] == [{"ApiKeyHeader": []}]


def test_http_upload_runs_through_real_storage_and_pyav_pipeline(tmp_path: Path) -> None:
    settings = make_settings().model_copy(update={"temp_storage_root": tmp_path})
    store = LocalEphemeralBlobStore(tmp_path)
    recognizer = PipelineRecognizer(store)
    service = BatchTranscriptionService(
        store=store,
        preprocessor=PyAvAudioPreprocessor(store),
        recognizer=recognizer,
        text_processor=NativeTranscriptTextProcessor(),
        max_upload_bytes=settings.max_upload_bytes,
        max_audio_duration_seconds=settings.max_audio_duration_seconds,
        sync_max_audio_duration_seconds=settings.sync_max_audio_duration_seconds,
        privacy_ttl_seconds=settings.privacy_ttl_seconds,
    )
    app = create_app(settings, transcription_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={"X-API-Key": TEST_API_KEY},
            files={"file": ("voice.wav", make_wav_bytes(), "audio/wav")},
            data={"language_mode": "explicit", "language": "en"},
        )

    assert response.status_code == 200
    assert response.json()["duration_seconds"] == pytest.approx(0.1)
    assert recognizer.normalized_bytes.startswith(b"RIFF")
    assert list(tmp_path.iterdir()) == []
