"""Secure batch-transcription orchestration tests."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionOutcome,
    BatchTranscriptionService,
    UploadCommand,
)
from speech_intelligence_api.domain.enums import (
    LanguageCode,
    LanguageSelectionMode,
    ProcessingMode,
)
from speech_intelligence_api.domain.errors import (
    AsyncProcessingRequiredError,
    AudioTooLongError,
    InvalidAudioError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from speech_intelligence_api.domain.models import (
    AudioProbe,
    BlobReference,
    LanguageSelection,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.audio import AudioPreprocessor
from speech_intelligence_api.ports.storage import EphemeralBlobStore
from speech_intelligence_api.ports.text import TranscriptTextProcessor


async def _chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


def _reference(
    key: str,
    *,
    media_type: str = "audio/wav",
    size_bytes: int = 4,
    expires_at: datetime | None = None,
) -> BlobReference:
    created_at = datetime.now(tz=UTC)
    return BlobReference(
        key=key,
        media_type=media_type,
        size_bytes=size_bytes,
        created_at=created_at,
        expires_at=expires_at or created_at + timedelta(minutes=5),
    )


def _result(text: str = "hello") -> TranscriptionResult:
    segment = TranscriptSegment(
        text=text,
        start_seconds=0,
        end_seconds=1,
        language=LanguageCode.ENGLISH,
    )
    return TranscriptionResult(
        language=LanguageCode.ENGLISH,
        language_confidence_estimate=0.95,
        text=text,
        segments=(segment,),
    )


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.put_calls = 0

    async def put(
        self,
        stream: AsyncIterable[bytes],
        *,
        media_type: str,
        expires_at: datetime,
    ) -> BlobReference:
        self.put_calls += 1
        contents = b"".join([chunk async for chunk in stream])
        reference = _reference(
            "raw.upload",
            media_type=media_type,
            size_bytes=len(contents),
            expires_at=expires_at,
        )
        self.objects[reference.key] = contents
        return reference

    async def read_chunks(self, reference: BlobReference) -> AsyncIterator[bytes]:
        yield self.objects[reference.key]

    async def delete(self, reference: BlobReference) -> bool:
        self.deleted.append(reference.key)
        return self.objects.pop(reference.key, None) is not None


class FakePreprocessor:
    def __init__(
        self,
        *,
        detected_media_type: str = "audio/wav",
        probe_duration: float | None = 10,
        normalized_duration: float = 10,
        failure: Exception | None = None,
    ) -> None:
        self.probe_value = AudioProbe(
            detected_media_type=detected_media_type,
            container_format="wav",
            codec="pcm_s16le",
            channels=1,
            sample_rate_hz=16_000,
            duration_seconds=probe_duration,
        )
        self.normalized_duration = normalized_duration
        self.failure = failure
        self.probed: BlobReference | None = None
        self.normalized: BlobReference | None = None

    async def probe(self, source: BlobReference) -> AudioProbe:
        self.probed = source
        if self.failure is not None:
            raise self.failure
        return self.probe_value

    async def normalize(
        self,
        source: BlobReference,
        *,
        expires_at: datetime,
        max_duration_seconds: float,
    ) -> tuple[BlobReference, float]:
        del source, max_duration_seconds
        self.normalized = _reference("normalized.wav", expires_at=expires_at)
        return self.normalized, self.normalized_duration


class FakeRecognizer:
    def __init__(self) -> None:
        self.request: TranscriptionRequest | None = None

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.request = request
        return _result()


class FakeTextProcessor:
    def __init__(self) -> None:
        self.called = False

    def process(self, result: TranscriptionResult) -> TranscriptionResult:
        self.called = True
        return result


def _service(
    store: MemoryStore,
    preprocessor: FakePreprocessor,
    recognizer: FakeRecognizer,
    text_processor: FakeTextProcessor,
    *,
    max_upload_bytes: int = 10,
    max_audio_duration_seconds: float = 100,
    sync_max_audio_duration_seconds: float = 20,
) -> BatchTranscriptionService:
    return BatchTranscriptionService(
        store=cast(EphemeralBlobStore, store),
        preprocessor=cast(AudioPreprocessor, preprocessor),
        recognizer=cast(SpeechRecognizer, recognizer),
        text_processor=cast(TranscriptTextProcessor, text_processor),
        max_upload_bytes=max_upload_bytes,
        max_audio_duration_seconds=max_audio_duration_seconds,
        sync_max_audio_duration_seconds=sync_max_audio_duration_seconds,
        privacy_ttl_seconds=300,
    )


def _command(
    *,
    filename: str = "speech.wav",
    media_type: str = "audio/wav",
    chunks: AsyncIterable[bytes] | None = None,
    processing_mode: ProcessingMode = ProcessingMode.AUTO,
    vocabulary: tuple[str, ...] = ("OpenAI",),
) -> UploadCommand:
    return UploadCommand(
        filename=filename,
        declared_media_type=media_type,
        chunks=chunks or _chunks(b"data"),
        language=LanguageSelection(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ENGLISH,
        ),
        processing_mode=processing_mode,
        vocabulary=vocabulary,
    )


@pytest.mark.asyncio
async def test_success_passes_options_and_always_deletes_audio() -> None:
    store = MemoryStore()
    preprocessor = FakePreprocessor()
    recognizer = FakeRecognizer()
    text_processor = FakeTextProcessor()
    service = _service(store, preprocessor, recognizer, text_processor)

    outcome = await service.execute(_command())

    assert isinstance(outcome, BatchTranscriptionOutcome)
    assert outcome.result.text == "hello"
    assert outcome.duration_seconds == 10
    assert recognizer.request is not None
    assert recognizer.request.audio.key == "normalized.wav"
    assert recognizer.request.vocabulary == ("OpenAI",)
    assert recognizer.request.word_timestamps is True
    assert text_processor.called is True
    assert store.deleted == ["normalized.wav", "raw.upload"]


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("", "audio/wav"),
        ("speech.txt", "audio/wav"),
        ("speech.wav", "text/plain"),
        ("speech.wav\x00", "audio/wav"),
        ("x" * 252 + ".wav", "audio/wav"),
    ],
)
@pytest.mark.asyncio
async def test_rejects_untrusted_upload_identity_before_storage(
    filename: str,
    media_type: str,
) -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    with pytest.raises(UnsupportedMediaTypeError):
        await service.execute(_command(filename=filename, media_type=media_type))

    assert store.put_calls == 0


@pytest.mark.asyncio
async def test_accepts_casefolded_extension_and_media_parameters() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    outcome = await service.execute(
        _command(filename=r"C:\fakepath\SPEECH.WAV", media_type="Audio/Wav; charset=binary")
    )

    assert isinstance(outcome, BatchTranscriptionOutcome)
    assert outcome.result.text == "hello"


@pytest.mark.asyncio
async def test_rejects_detected_container_mismatch_and_deletes_upload() -> None:
    store = MemoryStore()
    recognizer = FakeRecognizer()
    service = _service(
        store,
        FakePreprocessor(detected_media_type="audio/mpeg"),
        recognizer,
        FakeTextProcessor(),
    )

    with pytest.raises(UnsupportedMediaTypeError):
        await service.execute(_command())

    assert store.deleted == ["raw.upload"]
    assert recognizer.request is None


@pytest.mark.asyncio
async def test_rejects_probe_duration_over_total_limit() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(probe_duration=101),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    with pytest.raises(AudioTooLongError) as captured:
        await service.execute(_command())

    assert captured.value.details["max_duration_seconds"] == 100
    assert store.deleted == ["raw.upload"]


@pytest.mark.asyncio
async def test_decoded_long_audio_requires_async_and_cleans_both_files() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(probe_duration=None, normalized_duration=21),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    with pytest.raises(AsyncProcessingRequiredError):
        await service.execute(_command(processing_mode=ProcessingMode.SYNC))

    assert store.deleted == ["normalized.wav", "raw.upload"]


@pytest.mark.asyncio
async def test_explicit_async_mode_does_not_store_upload() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    with pytest.raises(AsyncProcessingRequiredError):
        await service.execute(_command(processing_mode=ProcessingMode.ASYNC))

    assert store.put_calls == 0


@pytest.mark.asyncio
async def test_chunk_limit_is_enforced_during_streaming() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(),
        FakeRecognizer(),
        FakeTextProcessor(),
        max_upload_bytes=5,
    )

    with pytest.raises(PayloadTooLargeError) as captured:
        await service.execute(_command(chunks=_chunks(b"123", b"", b"456")))

    assert captured.value.details["max_bytes"] == 5
    assert store.objects == {}


@pytest.mark.asyncio
async def test_preprocessor_failure_still_deletes_raw_upload() -> None:
    store = MemoryStore()
    service = _service(
        store,
        FakePreprocessor(failure=InvalidAudioError()),
        FakeRecognizer(),
        FakeTextProcessor(),
    )

    with pytest.raises(InvalidAudioError):
        await service.execute(_command())

    assert store.deleted == ["raw.upload"]


def test_upload_command_rejects_invalid_vocabulary() -> None:
    with pytest.raises(ValueError, match="100 entries"):
        _command(vocabulary=tuple("word" for _ in range(101)))
    with pytest.raises(ValueError, match="1 to 100"):
        _command(vocabulary=(" ",))
    with pytest.raises(ValueError, match="1 to 100"):
        _command(vocabulary=("x" * 101,))
