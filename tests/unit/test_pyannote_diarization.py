"""Local pyannote adapter tests without downloading gated model weights."""

from __future__ import annotations

import os
import struct
import sys
import wave
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import SecretStr

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.pyannote_diarization import PyannoteSpeakerDiarizer
from speech_intelligence_api.domain.errors import DiarizationUnavailableError
from speech_intelligence_api.domain.models import BlobReference
from tests.factories import make_settings


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


class Annotation:
    def __init__(self, entries: list[tuple[Segment, str]]) -> None:
        self.entries = entries

    def itertracks(self, *, yield_label: bool) -> list[tuple[Segment, None, str]]:
        assert yield_label is True
        return [(segment, None, label) for segment, label in self.entries]


@dataclass
class Output:
    speaker_diarization: object
    exclusive_speaker_diarization: object


class Pipeline:
    def __init__(self, output: object | Exception) -> None:
        self.output = output
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, audio: object, **kwargs: object) -> object:
        self.calls.append((audio, kwargs))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _blob() -> BlobReference:
    now = datetime.now(tz=UTC)
    return BlobReference("audio.wav", "audio/wav", 4, now, now + timedelta(minutes=5))


def _path_loader(path: Path) -> str:
    return str(path)


@pytest.mark.asyncio
async def test_maps_exclusive_turns_and_detects_regular_overlap(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    output = Output(
        speaker_diarization=Annotation(
            [
                (Segment(0, 1.2), "a"),
                (Segment(0.8, 1.5), "b"),
            ]
        ),
        exclusive_speaker_diarization=Annotation(
            [
                (Segment(0, 1), "a"),
                (Segment(1, 1.5), "b"),
            ]
        ),
    )
    pipeline = Pipeline(output)
    loads = 0

    def factory() -> Pipeline:
        nonlocal loads
        loads += 1
        return pipeline

    adapter = PyannoteSpeakerDiarizer(
        make_settings(),
        LocalEphemeralBlobStore(tmp_path),
        pipeline_factory=factory,
        waveform_loader=_path_loader,
    )

    first = await adapter.diarize(_blob(), expected_speakers=2)
    second = await adapter.diarize(_blob())

    assert loads == 1
    assert pipeline.calls[0][1] == {"num_speakers": 2}
    assert pipeline.calls[1][1] == {}
    assert [(turn.speaker, turn.overlapping_speech) for turn in first] == [
        ("a", True),
        ("b", True),
    ]
    assert second == first


def test_supports_plain_iterable_annotations_and_sorts_turns() -> None:
    output = Output(
        speaker_diarization=[(Segment(0, 1), "a")],
        exclusive_speaker_diarization=[
            (Segment(1, 2), "b"),
            (Segment(0, 1), "a"),
            (Segment(2, 2), "ignored"),
        ],
    )

    turns = PyannoteSpeakerDiarizer._turns(output)

    assert [(turn.start_seconds, turn.speaker) for turn in turns] == [(0, "a"), (1, "b")]


@pytest.mark.asyncio
async def test_sanitizes_pipeline_and_malformed_output_failures(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    store = LocalEphemeralBlobStore(tmp_path)
    failing = PyannoteSpeakerDiarizer(
        make_settings(),
        store,
        pipeline_factory=lambda: Pipeline(RuntimeError("private")),
        waveform_loader=_path_loader,
    )
    malformed = PyannoteSpeakerDiarizer(
        make_settings(),
        store,
        pipeline_factory=lambda: Pipeline(object()),
        waveform_loader=_path_loader,
    )

    with pytest.raises(DiarizationUnavailableError):
        await failing.diarize(_blob())
    with pytest.raises(DiarizationUnavailableError):
        await malformed.diarize(_blob())


def test_loads_normalized_pcm16_waveform_without_file_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.operations: list[tuple[object, ...]] = []

        def numel(self) -> int:
            return 4

        def reshape(self, *shape: int) -> FakeTensor:
            self.operations.append(("reshape", *shape))
            return self

        def transpose(self, first: int, second: int) -> FakeTensor:
            self.operations.append(("transpose", first, second))
            return self

        def contiguous(self) -> FakeTensor:
            self.operations.append(("contiguous",))
            return self

        def to(self, *, dtype: object) -> FakeTensor:
            self.operations.append(("to", dtype))
            return self

        def div_(self, divisor: float) -> FakeTensor:
            self.operations.append(("div", divisor))
            return self

    tensor = FakeTensor()
    torch = ModuleType("torch")
    torch.int16 = object()  # type: ignore[attr-defined]
    torch.float32 = object()  # type: ignore[attr-defined]
    captured: dict[str, object] = {}

    def frombuffer(buffer: bytearray, *, dtype: object) -> FakeTensor:
        captured.update(buffer=bytes(buffer), dtype=dtype)
        return tensor

    torch.frombuffer = frombuffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    path = tmp_path / "normalized.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack("<hhhh", 0, 32767, -32768, 1))

    payload = PyannoteSpeakerDiarizer._waveform_payload(path)

    assert payload == {"waveform": tensor, "sample_rate": 16_000}
    assert captured == {
        "buffer": struct.pack("<hhhh", 0, 32767, -32768, 1),
        "dtype": torch.int16,
    }
    assert tensor.operations == [
        ("reshape", -1, 2),
        ("transpose", 0, 1),
        ("contiguous",),
        ("to", torch.float32),
        ("div", 32768.0),
    ]


def test_local_only_configuration_rejects_missing_model_directory(tmp_path: Path) -> None:
    settings = make_settings().model_copy(
        update={
            "diarization_model_source": str(tmp_path / "missing"),
            "diarization_model_local_files_only": True,
        }
    )
    adapter = PyannoteSpeakerDiarizer(settings, LocalEphemeralBlobStore(tmp_path))

    with pytest.raises(DiarizationUnavailableError):
        adapter._load_pipeline()


def test_loads_gated_pipeline_and_moves_it_to_configured_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoadedPipeline:
        def __init__(self) -> None:
            self.device: object | None = None

        def __call__(self, audio: object, **kwargs: object) -> object:
            return object()

        def to(self, device: object) -> None:
            self.device = device

    loaded = LoadedPipeline()

    class PipelineLoader:
        source: str | None = None
        token: str | None = None

        @classmethod
        def from_pretrained(cls, source: str, *, token: str | None) -> LoadedPipeline:
            cls.source = source
            cls.token = token
            return loaded

    pyannote = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = PipelineLoader  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.device = lambda name: f"device:{name}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")
    settings = make_settings().model_copy(
        update={
            "diarization_device": "cuda",
            "diarization_huggingface_token": SecretStr("read-only-token"),
        }
    )
    adapter = PyannoteSpeakerDiarizer(settings, LocalEphemeralBlobStore(tmp_path))

    pipeline = adapter._load_pipeline()

    assert pipeline is loaded
    assert PipelineLoader.source == "pyannote/speaker-diarization-community-1"
    assert PipelineLoader.token == "read-only-token"
    assert loaded.device == "device:cuda"
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"


@pytest.mark.asyncio
async def test_sanitizes_pipeline_factory_loading_failure(tmp_path: Path) -> None:
    def fail() -> Pipeline:
        raise RuntimeError("private model path")

    adapter = PyannoteSpeakerDiarizer(
        make_settings(),
        LocalEphemeralBlobStore(tmp_path),
        pipeline_factory=fail,
    )

    with pytest.raises(DiarizationUnavailableError):
        await adapter._get_pipeline()

    unavailable = PyannoteSpeakerDiarizer(
        make_settings(),
        LocalEphemeralBlobStore(tmp_path),
        pipeline_factory=lambda: (_ for _ in ()).throw(DiarizationUnavailableError()),
    )
    with pytest.raises(DiarizationUnavailableError):
        await unavailable._get_pipeline()


def test_sanitizes_optional_pyannote_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyannote = ModuleType("pyannote")
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.delitem(sys.modules, "pyannote.audio", raising=False)
    adapter = PyannoteSpeakerDiarizer(
        make_settings(),
        LocalEphemeralBlobStore(tmp_path),
    )

    with pytest.raises(DiarizationUnavailableError):
        adapter._load_pipeline()
