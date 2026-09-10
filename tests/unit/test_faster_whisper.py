"""Faster-Whisper adapter contract tests without downloading model weights."""

from __future__ import annotations

import math
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from speech_intelligence_api.adapters.faster_whisper import (
    FasterWhisperSpeechRecognizer,
    ModelFactory,
)
from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.domain.enums import (
    ChineseScript,
    LanguageCode,
    LanguageSelectionMode,
)
from speech_intelligence_api.domain.errors import (
    InvalidAudioError,
    ModelUnavailableError,
    UncertainLanguageError,
)
from speech_intelligence_api.domain.models import (
    BlobReference,
    LanguageSelection,
    TranscriptionRequest,
)
from tests.factories import make_settings


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    words: list[FakeWord] | None
    avg_logprob: float


@dataclass
class FakeInfo:
    language: str
    language_probability: float
    all_language_probs: list[tuple[str, float]] | None = None


Detection = tuple[str, float, list[tuple[str, float]]]


class FakeModel:
    def __init__(
        self,
        outcomes: list[tuple[Iterable[FakeSegment], FakeInfo] | Exception],
        detections: list[Detection] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.detections = detections or []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.detect_calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: str,
        **kwargs: Any,
    ) -> tuple[Iterable[FakeSegment], FakeInfo]:
        self.calls.append((audio, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def detect_language(self, **kwargs: Any) -> Detection:
        self.detect_calls.append(kwargs)
        return self.detections.pop(0)


def _audio_reference() -> BlobReference:
    created_at = datetime.now(tz=UTC)
    return BlobReference(
        key="normalized.wav",
        media_type="audio/wav",
        size_bytes=1024,
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=5),
    )


def _request(
    *,
    mode: LanguageSelectionMode,
    language: LanguageCode | None = None,
    chinese_script: ChineseScript | None = None,
    vocabulary: tuple[str, ...] = (),
    word_timestamps: bool = True,
    draft: bool = False,
) -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=_audio_reference(),
        language=LanguageSelection(
            mode=mode,
            language=language,
            chinese_script=chinese_script,
        ),
        vocabulary=vocabulary,
        word_timestamps=word_timestamps,
        draft=draft,
    )


def _recognizer(
    tmp_path: Path,
    model: FakeModel,
    *,
    threshold: float = 0.7,
    draft_model_name: str | None = None,
    loaded_names: list[str] | None = None,
) -> FasterWhisperSpeechRecognizer:
    settings = make_settings().model_copy(
        update={
            "temp_storage_root": tmp_path,
            "language_confidence_threshold": threshold,
            "asr_draft_model_name": draft_model_name,
        }
    )

    def factory(model_name: str) -> FakeModel:
        if loaded_names is not None:
            loaded_names.append(model_name)
        return model

    return FasterWhisperSpeechRecognizer(
        settings,
        LocalEphemeralBlobStore(tmp_path),
        model_factory=cast(ModelFactory, factory),
        # FakeModel scores whatever it is handed, so the loader stays a stub
        # rather than pulling faster-whisper's decoder into unit tests.
        audio_loader=lambda audio_path: audio_path,
    )


def _speech_segment(text: str = " مرحبا ") -> FakeSegment:
    return FakeSegment(
        text=text,
        start=-0.1,
        end=1.0,
        words=[
            FakeWord(word=" مرحبا ", start=-1, end=2, probability=1.2),
            FakeWord(word=" ", start=0, end=1, probability=0.5),
        ],
        avg_logprob=-0.2,
    )


@pytest.mark.asyncio
async def test_explicit_language_transcribes_in_source_language(tmp_path: Path) -> None:
    model = FakeModel([([_speech_segment()], FakeInfo(language="ar", language_probability=0.99))])
    recognizer = _recognizer(tmp_path, model)

    result = await recognizer.transcribe(
        _request(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ARABIC,
            vocabulary=("OpenAI", "واجهة"),
        )
    )

    assert result.language is LanguageCode.ARABIC
    assert result.language_confidence_estimate == 1
    assert result.text == "مرحبا"
    assert result.segments[0].start_seconds == 0
    assert result.segments[0].words[0].start_seconds == 0
    assert result.segments[0].words[0].end_seconds == 1
    assert result.segments[0].words[0].confidence_estimate == 1
    assert result.segments[0].confidence_estimate == pytest.approx(math.exp(-0.2))
    assert len(model.calls) == 1
    options = model.calls[0][1]
    assert options["language"] == "ar"
    assert options["task"] == "transcribe"
    assert options["hotwords"] == "OpenAI, واجهة"
    assert options["word_timestamps"] is True
    assert options["vad_filter"] is True
    assert options["beam_size"] == 5
    assert options["condition_on_previous_text"] is True


@pytest.mark.asyncio
async def test_draft_requests_decode_greedily_for_live_partial_latency(tmp_path: Path) -> None:
    model = FakeModel([([_speech_segment()], FakeInfo(language="en", language_probability=0.99))])
    recognizer = _recognizer(tmp_path, model)

    await recognizer.transcribe(
        _request(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ENGLISH,
            word_timestamps=True,
            draft=True,
        )
    )

    options = model.calls[0][1]
    assert options["beam_size"] == 1
    assert options["condition_on_previous_text"] is False
    assert options["word_timestamps"] is False


@pytest.mark.asyncio
async def test_draft_requests_use_the_configured_draft_model(tmp_path: Path) -> None:
    model = FakeModel(
        [
            ([_speech_segment()], FakeInfo(language="en", language_probability=0.99)),
            ([_speech_segment()], FakeInfo(language="en", language_probability=0.99)),
        ]
    )
    loaded_names: list[str] = []
    recognizer = _recognizer(tmp_path, model, draft_model_name="tiny", loaded_names=loaded_names)

    await recognizer.transcribe(
        _request(mode=LanguageSelectionMode.EXPLICIT, language=LanguageCode.ENGLISH, draft=True)
    )
    await recognizer.transcribe(
        _request(mode=LanguageSelectionMode.EXPLICIT, language=LanguageCode.ENGLISH)
    )

    # A final keeps the accurate main model; only the replaceable partial downgrades.
    assert loaded_names == ["tiny", make_settings().asr_model_name]


@pytest.mark.asyncio
async def test_automatic_language_filters_to_supported_allowlist(tmp_path: Path) -> None:
    model = FakeModel(
        [([_speech_segment(" bonjour ")], FakeInfo("fr", 0.82))],
        detections=[("xx", 0.99, [("xx", 0.99), ("fr", 0.82), ("de", 0.11)])],
    )
    recognizer = _recognizer(tmp_path, model)

    result = await recognizer.transcribe(_request(mode=LanguageSelectionMode.AUTOMATIC))

    assert result.language is LanguageCode.FRENCH
    assert result.language_confidence_estimate == pytest.approx(0.82)
    # Detection reads the encoder only, so automatic mode costs one decode, not two.
    assert len(model.calls) == 1
    detection_options = model.detect_calls[0]
    assert detection_options["vad_filter"] is True
    assert detection_options["language_detection_threshold"] == 1.0
    assert detection_options["language_detection_segments"] == 3
    assert model.calls[0][1]["language"] == "fr"
    assert model.calls[0][1]["task"] == "transcribe"


@pytest.mark.asyncio
async def test_automatic_language_rejects_low_confidence(tmp_path: Path) -> None:
    model = FakeModel([], detections=[("en", 0.5, [("en", 0.5), ("fr", 0.4)])])
    recognizer = _recognizer(tmp_path, model, threshold=0.75)

    with pytest.raises(UncertainLanguageError) as captured:
        await recognizer.transcribe(_request(mode=LanguageSelectionMode.AUTOMATIC))

    assert captured.value.details["confidence_threshold"] == 0.75
    assert captured.value.details["candidates"][0]["language"] == "en"
    # An uncertain language must not spend a decode pass.
    assert model.calls == []


@pytest.mark.asyncio
async def test_automatic_language_rejects_no_supported_candidates(tmp_path: Path) -> None:
    model = FakeModel([], detections=[("xx", 0.99, [("xx", 0.99)])])
    recognizer = _recognizer(tmp_path, model)

    with pytest.raises(UncertainLanguageError) as captured:
        await recognizer.transcribe(_request(mode=LanguageSelectionMode.AUTOMATIC))

    assert captured.value.details["candidates"] == []


@pytest.mark.asyncio
async def test_chinese_uses_configured_default_script(tmp_path: Path) -> None:
    model = FakeModel(
        [([_speech_segment("漢語")], FakeInfo("zh", 0.95))],
        detections=[("zh", 0.95, [("zh", 0.95)])],
    )
    recognizer = _recognizer(tmp_path, model)

    result = await recognizer.transcribe(_request(mode=LanguageSelectionMode.AUTOMATIC))

    assert result.language is LanguageCode.CHINESE
    assert result.chinese_script is ChineseScript.SIMPLIFIED


@pytest.mark.asyncio
async def test_blank_segments_are_rejected_as_no_speech(tmp_path: Path) -> None:
    model = FakeModel(
        [
            (
                [FakeSegment(" ", 0, 1, None, -1)],
                FakeInfo("en", 1),
            )
        ]
    )
    recognizer = _recognizer(tmp_path, model)

    with pytest.raises(InvalidAudioError) as captured:
        await recognizer.transcribe(
            _request(
                mode=LanguageSelectionMode.EXPLICIT,
                language=LanguageCode.ENGLISH,
            )
        )

    assert "No speech" in captured.value.public_message


@pytest.mark.asyncio
async def test_model_load_failure_is_sanitized(tmp_path: Path) -> None:
    settings = make_settings().model_copy(update={"temp_storage_root": tmp_path})

    def fail_factory() -> Any:
        raise RuntimeError("secret provider detail")

    recognizer = FasterWhisperSpeechRecognizer(
        settings,
        LocalEphemeralBlobStore(tmp_path),
        model_factory=cast(ModelFactory, fail_factory),
    )

    with pytest.raises(ModelUnavailableError) as captured:
        await recognizer.transcribe(
            _request(
                mode=LanguageSelectionMode.EXPLICIT,
                language=LanguageCode.ENGLISH,
            )
        )
    assert "secret provider detail" not in captured.value.public_message


@pytest.mark.asyncio
async def test_inference_failure_is_sanitized_and_model_is_loaded_once(tmp_path: Path) -> None:
    model = FakeModel(
        [
            RuntimeError("secret inference detail"),
            ([_speech_segment("hello")], FakeInfo("en", 1)),
        ]
    )
    factory_calls = 0

    def factory(model_name: str) -> Any:
        nonlocal factory_calls
        factory_calls += 1
        return model

    settings = make_settings().model_copy(update={"temp_storage_root": tmp_path})
    recognizer = FasterWhisperSpeechRecognizer(
        settings,
        LocalEphemeralBlobStore(tmp_path),
        model_factory=cast(ModelFactory, factory),
    )
    request = _request(
        mode=LanguageSelectionMode.EXPLICIT,
        language=LanguageCode.ENGLISH,
    )

    with pytest.raises(ModelUnavailableError):
        await recognizer.transcribe(request)
    result = await recognizer.transcribe(request)

    assert result.text == "hello"
    assert factory_calls == 1


def test_default_model_factory_passes_production_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import faster_whisper  # type: ignore[import-untyped]

    captured: dict[str, Any] = {}
    expected_model = object()

    def fake_whisper_model(name: str, **kwargs: Any) -> object:
        captured["name"] = name
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(faster_whisper, "WhisperModel", fake_whisper_model)
    download_root = tmp_path / "models"
    settings = make_settings().model_copy(
        update={
            "temp_storage_root": tmp_path,
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_cpu_threads": 2,
            "asr_num_workers": 3,
            "asr_model_download_root": download_root,
            "asr_model_local_files_only": True,
        }
    )
    recognizer = FasterWhisperSpeechRecognizer(
        settings,
        LocalEphemeralBlobStore(tmp_path),
    )

    model = recognizer._build_model(settings.asr_model_name)

    assert model is expected_model
    assert captured == {
        "name": "large-v3",
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 2,
        "num_workers": 3,
        "download_root": str(download_root),
        "local_files_only": True,
    }


def test_default_audio_loader_resamples_to_the_detection_rate(tmp_path: Path) -> None:
    """Whisper's feature extractor is fixed at 16 kHz.

    Handing it audio at the file's own rate would shift every mel frame and
    quietly skew language detection, so the loader has to resample rather than
    just read.
    """

    source = tmp_path / "tone.wav"
    source_rate = 8000
    frames = source_rate // 2
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(source_rate)
        handle.writeframes(bytes([0x00, 0x10]) * frames)

    audio = FasterWhisperSpeechRecognizer._decode_audio(str(source))

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.shape[0] == pytest.approx(frames * 2, rel=0.05)
