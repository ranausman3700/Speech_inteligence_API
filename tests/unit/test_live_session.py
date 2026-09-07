"""Transport-neutral real-time dictation session tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from speech_intelligence_api.adapters.native_text import NativeTranscriptTextProcessor
from speech_intelligence_api.application.live_transcription import LiveTranscriptionService
from speech_intelligence_api.domain.enums import (
    LanguageCode,
    LanguageSelectionMode,
    LiveEventType,
)
from speech_intelligence_api.domain.errors import (
    LiveCapacityExceededError,
    LiveSessionLimitError,
)
from speech_intelligence_api.domain.models import (
    BlobReference,
    LanguageSelection,
    LiveTranscriptionOptions,
    SpeechRegion,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)


class SequenceDetector:
    def __init__(self, responses: list[tuple[SpeechRegion, ...]]) -> None:
        self.responses = responses

    async def detect(self, pcm_audio: bytes, *, sample_rate_hz: int) -> tuple[SpeechRegion, ...]:
        assert pcm_audio
        assert sample_rate_hz == 16_000
        return self.responses.pop(0) if self.responses else ()


class MemorySnapshots:
    def __init__(self) -> None:
        self.created: list[bytes] = []
        self.deleted: list[str] = []

    async def create(
        self,
        pcm_audio: bytes,
        *,
        sample_rate_hz: int,
        expires_at: datetime,
    ) -> BlobReference:
        self.created.append(pcm_audio)
        return BlobReference(
            key=f"snapshot-{len(self.created)}.wav",
            media_type="audio/wav",
            size_bytes=len(pcm_audio),
            created_at=datetime.now(tz=UTC),
            expires_at=expires_at,
        )

    async def delete(self, reference: BlobReference) -> bool:
        self.deleted.append(reference.key)
        return True


class RecordingRecognizer:
    def __init__(self) -> None:
        self.requests: list[TranscriptionRequest] = []

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.requests.append(request)
        segment = TranscriptSegment(
            text=" hello ",
            start_seconds=0,
            end_seconds=0.02,
            language=LanguageCode.ENGLISH,
        )
        return TranscriptionResult(
            language=LanguageCode.ENGLISH,
            language_confidence_estimate=1,
            text=" hello ",
            segments=(segment,),
        )


def _options() -> LiveTranscriptionOptions:
    return LiveTranscriptionOptions(
        language=LanguageSelection(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ENGLISH,
        ),
        vocabulary=("FastAPI",),
    )


def _service(
    detector: SequenceDetector,
    snapshots: MemorySnapshots,
    recognizer: RecordingRecognizer,
    *,
    max_sessions: int = 1,
    max_session_seconds: float = 1,
) -> LiveTranscriptionService:
    return LiveTranscriptionService(
        detector=detector,
        snapshot_store=snapshots,
        recognizer=recognizer,
        text_processor=NativeTranscriptTextProcessor(),
        max_sessions=max_sessions,
        max_session_seconds=max_session_seconds,
        max_utterance_seconds=0.5,
        max_chunk_bytes=4096,
        end_silence_ms=32,
        analysis_window_seconds=0.25,
        pre_roll_ms=64,
        partial_min_audio_seconds=0.032,
        partial_interval_seconds=0.032,
        privacy_ttl_seconds=60,
    )


@pytest.mark.asyncio
async def test_live_session_emits_partial_and_final_with_snapshot_cleanup() -> None:
    speech = (SpeechRegion(0, 0.032, 0.9),)
    snapshots = MemorySnapshots()
    recognizer = RecordingRecognizer()
    service = _service(SequenceDetector([speech, ()]), snapshots, recognizer)
    frame = b"\x01\x00" * 512

    async with service.session(_options()) as session:
        first = await session.ingest(frame)
        second = await session.ingest(frame)

    assert [event.event_type for event in first] == [
        LiveEventType.SPEECH_STARTED,
        LiveEventType.PARTIAL,
    ]
    assert [event.event_type for event in second] == [LiveEventType.FINAL]
    assert first[1].result is not None
    assert first[1].result.text == "hello"
    assert second[0].result is not None
    assert second[0].result.text == "hello"
    assert len(snapshots.created) == 2
    assert snapshots.deleted == ["snapshot-1.wav", "snapshot-2.wav"]
    assert recognizer.requests[0].vocabulary == ("FastAPI",)


@pytest.mark.asyncio
async def test_live_service_enforces_session_capacity() -> None:
    service = _service(SequenceDetector([]), MemorySnapshots(), RecordingRecognizer())

    async with service.session(_options()):
        with pytest.raises(LiveCapacityExceededError):
            async with service.session(_options()):
                pytest.fail("capacity-exceeded session was admitted")


@pytest.mark.asyncio
async def test_live_session_rejects_invalid_frames_and_duration_overflow() -> None:
    service = _service(
        SequenceDetector([()]),
        MemorySnapshots(),
        RecordingRecognizer(),
        max_session_seconds=0.032,
    )

    async with service.session(_options()) as session:
        with pytest.raises(ValueError, match="invalid"):
            await session.ingest(b"x")
        await session.ingest(bytes(512 * 2))
        with pytest.raises(LiveSessionLimitError):
            await session.ingest(bytes(2))


@pytest.mark.asyncio
async def test_live_session_commit_without_speech_and_close_are_idempotent() -> None:
    service = _service(SequenceDetector([]), MemorySnapshots(), RecordingRecognizer())

    async with service.session(_options()) as session:
        assert await session.ingest(b"") == ()
        assert await session.commit() == ()
        assert await session.close() == ()
        assert await session.close() == ()
        with pytest.raises(RuntimeError, match="closed"):
            await session.ingest(b"\0\0")
