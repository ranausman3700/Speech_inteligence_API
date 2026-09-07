"""Authenticated live-transcription WebSocket protocol tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from speech_intelligence_api.application.live_transcription import (
    LiveTranscriptionService,
)
from speech_intelligence_api.application.transcriptions import BatchTranscriptionService
from speech_intelligence_api.domain.enums import LanguageCode, LiveEventType
from speech_intelligence_api.domain.errors import (
    InvalidAudioError,
    LiveCapacityExceededError,
    ModelUnavailableError,
    ServiceError,
)
from speech_intelligence_api.domain.models import (
    LiveSessionEvent,
    LiveTranscriptionOptions,
    TranscriptionResult,
    TranscriptSegment,
)
from speech_intelligence_api.entrypoints.http.app import create_app
from speech_intelligence_api.ports.rate_limiting import RateLimitDecision, RateLimiter
from tests.factories import TEST_API_KEY, make_settings


class FakeLiveSession:
    def __init__(self, failure: ServiceError | None = None) -> None:
        self.received = bytearray()
        self.options: LiveTranscriptionOptions | None = None
        self.closed = False
        self.discarded = False
        self._utterance_active = False
        self.failure = failure

    async def ingest(self, pcm_chunk: bytes) -> tuple[LiveSessionEvent, ...]:
        if self.failure is not None:
            raise self.failure
        self.received.extend(pcm_chunk)
        self._utterance_active = True
        return (
            LiveSessionEvent(
                event_type=LiveEventType.SPEECH_STARTED,
                utterance_id=1,
                revision=0,
                start_seconds=0,
                end_seconds=0,
            ),
            self._transcript(LiveEventType.PARTIAL, 1),
        )

    async def commit(self) -> tuple[LiveSessionEvent, ...]:
        if not self._utterance_active:
            return ()
        self._utterance_active = False
        return (self._transcript(LiveEventType.FINAL, 2),)

    async def close(self) -> tuple[LiveSessionEvent, ...]:
        events = await self.commit()
        self.closed = True
        return events

    async def discard(self) -> None:
        self.discarded = True
        self.received.clear()

    @staticmethod
    def _transcript(event_type: LiveEventType, revision: int) -> LiveSessionEvent:
        result = TranscriptionResult(
            language=LanguageCode.ENGLISH,
            language_confidence_estimate=1,
            text="Hello.",
            segments=(
                TranscriptSegment(
                    text="Hello.",
                    start_seconds=0,
                    end_seconds=0.5,
                    language=LanguageCode.ENGLISH,
                    confidence_estimate=0.95,
                ),
            ),
        )
        return LiveSessionEvent(
            event_type=event_type,
            utterance_id=1,
            revision=revision,
            start_seconds=0,
            end_seconds=0.5,
            result=result,
        )


class FakeLiveService:
    def __init__(
        self,
        *,
        session_failure: ServiceError | None = None,
        admission_failure: ServiceError | None = None,
    ) -> None:
        self.session_instance = FakeLiveSession(session_failure)
        self.options: LiveTranscriptionOptions | None = None
        self.admission_failure = admission_failure

    @asynccontextmanager
    async def session(
        self,
        options: LiveTranscriptionOptions,
    ) -> AsyncIterator[FakeLiveSession]:
        if self.admission_failure is not None:
            raise self.admission_failure
        self.options = options
        self.session_instance.options = options
        try:
            yield self.session_instance
        finally:
            await self.session_instance.discard()


class _DenyingRateLimiter:
    async def acquire(
        self,
        identity: str,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=False,
            limit=limit,
            remaining=0,
            retry_after_seconds=9,
            reset_after_seconds=9,
        )

    async def ping(self) -> None:
        return None


def _client(
    service: FakeLiveService,
    *,
    max_chunk_bytes: int = 32_000,
    cors_allowed_origins: tuple[str, ...] = (),
    rate_limiter: RateLimiter | None = None,
) -> TestClient:
    settings = make_settings().model_copy(
        update={
            "live_max_chunk_bytes": max_chunk_bytes,
            "cors_allowed_origins": cors_allowed_origins,
            "rate_limit_enabled": rate_limiter is not None,
            "rate_limit_live_sessions": 1,
        }
    )
    app = create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, object()),
        live_transcription_service=cast(LiveTranscriptionService, service),
        rate_limiter=rate_limiter,
    )
    return TestClient(app)


def _start_message(*, include_key: bool = False) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "start",
        "encoding": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "language_mode": "explicit",
        "language": "en",
        "vocabulary": ["FastAPI"],
        "word_timestamps": True,
    }
    if include_key:
        message["api_key"] = TEST_API_KEY
    return message


def test_live_websocket_emits_ready_partial_final_and_orderly_close() -> None:
    service = FakeLiveService()

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY, "X-Request-ID": "live-request-1"},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        ready = websocket.receive_json()
        websocket.send_bytes(b"\x01\x00" * 512)
        speech_started = websocket.receive_json()
        partial = websocket.receive_json()
        websocket.send_json({"type": "ping"})
        pong = websocket.receive_json()
        websocket.send_json({"type": "stop"})
        final = websocket.receive_json()
        closed = websocket.receive_json()

    assert ready["type"] == "ready"
    assert ready["request_id"] == "live-request-1"
    assert ready["encoding"] == "pcm_s16le"
    assert speech_started["type"] == "speech_started"
    assert partial["type"] == "partial"
    assert partial["text"] == "Hello."
    assert pong["type"] == "pong"
    assert final["type"] == "final"
    assert final["segments"][0]["confidence_estimate"] == 0.95
    assert closed["type"] == "session_closed"
    assert service.options is not None
    assert service.options.vocabulary == ("FastAPI",)
    assert service.session_instance.discarded is True


def test_live_websocket_supports_browser_compatible_start_message_authentication() -> None:
    service = FakeLiveService()

    with (
        _client(service) as client,
        client.websocket_connect("/v1/live-transcription") as websocket,
    ):
        websocket.send_json(_start_message(include_key=True))
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "stop"})
        assert websocket.receive_json()["type"] == "session_closed"


def test_live_websocket_rejects_missing_key_without_echoing_credentials() -> None:
    service = FakeLiveService()

    with (
        _client(service) as client,
        client.websocket_connect("/v1/live-transcription") as websocket,
    ):
        websocket.send_json(_start_message())
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "authentication_failed"
        assert error["fatal"] is True
        with pytest.raises(WebSocketDisconnect) as exc_info:
            websocket.receive_json()

    assert exc_info.value.code == 4401
    assert service.options is None


def test_live_websocket_enforces_distributed_session_rate_limit() -> None:
    service = FakeLiveService()

    with (
        _client(service, rate_limiter=_DenyingRateLimiter()) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        error = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()

    assert error["code"] == "rate_limited"
    assert error["details"] == {
        "limit": 1,
        "window_seconds": 60,
        "retry_after_seconds": 9,
    }
    assert error["fatal"] is True
    assert closed.value.code == 4429
    assert service.options is None


def test_live_websocket_rejects_oversized_or_partial_pcm_frames() -> None:
    service = FakeLiveService()

    with (
        _client(service, max_chunk_bytes=1024) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(bytes(1026))
        error = websocket.receive_json()
        assert error["code"] == "payload_too_large"
        assert error["fatal"] is True

    assert service.session_instance.received == b""


def test_live_websocket_rejects_cross_origin_handshake() -> None:
    service = FakeLiveService()

    with (
        _client(service, cors_allowed_origins=("https://trusted.example",)) as client,
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"Origin": "https://untrusted.example"},
        ),
    ):
        pytest.fail("cross-origin WebSocket was accepted")

    assert exc_info.value.code == 4403


def test_capabilities_advertise_live_pcm_contract() -> None:
    service = FakeLiveService()

    with _client(service) as client:
        response = client.get("/v1/capabilities", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 200
    live = response.json()["live_transcription"]
    assert live["websocket_path"] == "/v1/live-transcription"
    assert live["sample_rate_hz"] == 16_000
    assert live["vad"] == "silero"
    assert {"partial", "final", "error"} <= set(live["server_events"])


@pytest.mark.parametrize(
    "first_message",
    [b"binary-is-not-a-start-message", "{not-json"],
)
def test_live_websocket_rejects_invalid_first_message(first_message: bytes | str) -> None:
    service = FakeLiveService()

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        if isinstance(first_message, bytes):
            websocket.send_bytes(first_message)
        else:
            websocket.send_text(first_message)
        error = websocket.receive_json()

    assert error["code"] == "invalid_request"
    assert error["fatal"] is True
    assert service.options is None


def test_live_websocket_rejects_invalid_language_combination() -> None:
    service = FakeLiveService()
    message = _start_message()
    message.pop("language")

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(message)
        error = websocket.receive_json()

    assert error["code"] == "invalid_request"
    assert "options" in error["detail"]


def test_live_websocket_reports_disabled_or_unavailable_service() -> None:
    disabled_settings = make_settings().model_copy(update={"live_transcription_enabled": False})
    disabled_app = create_app(
        disabled_settings,
        transcription_service=cast(BatchTranscriptionService, object()),
    )
    unavailable_app = create_app(
        make_settings(),
        transcription_service=cast(BatchTranscriptionService, object()),
    )

    with (
        TestClient(disabled_app) as client,
        pytest.raises(WebSocketDisconnect) as disabled,
        client.websocket_connect("/v1/live-transcription"),
    ):
        pytest.fail("disabled live service was accepted")
    with (
        TestClient(unavailable_app) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        error = websocket.receive_json()

    assert disabled.value.code == 1011
    assert error["code"] == "dependency_unavailable"
    assert error["fatal"] is True


def test_live_websocket_reports_capacity_backpressure() -> None:
    service = FakeLiveService(admission_failure=LiveCapacityExceededError(1))

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        error = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()

    assert error["code"] == "capacity_exceeded"
    assert error["details"] == {"max_sessions": 1}
    assert closed.value.code == 4429


def test_live_websocket_keeps_session_open_after_recoverable_utterance_error() -> None:
    service = FakeLiveService(session_failure=InvalidAudioError())

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(bytes(1024))
        error = websocket.receive_json()
        websocket.send_json({"type": "ping"})
        pong = websocket.receive_json()
        websocket.send_json({"type": "stop"})
        closed = websocket.receive_json()

    assert error["code"] == "invalid_audio"
    assert error["fatal"] is False
    assert pong["type"] == "pong"
    assert closed["type"] == "session_closed"


def test_live_websocket_closes_after_model_failure() -> None:
    service = FakeLiveService(session_failure=ModelUnavailableError())

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(bytes(1024))
        error = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()

    assert error["code"] == "model_unavailable"
    assert error["fatal"] is True
    assert closed.value.code == 1011


def test_live_websocket_rejects_invalid_control_and_partial_pcm() -> None:
    service = FakeLiveService()

    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json({"type": "unknown"})
        invalid_control = websocket.receive_json()

    service = FakeLiveService()
    with (
        _client(service) as client,
        client.websocket_connect(
            "/v1/live-transcription",
            headers={"X-API-Key": TEST_API_KEY},
        ) as websocket,
    ):
        websocket.send_json(_start_message())
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(b"x")
        partial_pcm = websocket.receive_json()

    assert invalid_control["code"] == "invalid_request"
    assert partial_pcm["code"] == "invalid_request"
