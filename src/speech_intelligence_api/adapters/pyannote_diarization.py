"""Local pyannote Community-1 speaker diarization adapter."""

from __future__ import annotations

import asyncio
import os
import time
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import anyio

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.observability import NoopObservability
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.errors import DiarizationUnavailableError
from speech_intelligence_api.domain.models import BlobReference, DiarizationTurn
from speech_intelligence_api.ports.observability import Observability


class DiarizationPipeline(Protocol):
    """Dynamic pyannote pipeline subset kept out of the core dependency graph."""

    def __call__(self, audio: object, **kwargs: object) -> object: ...


PipelineFactory = Callable[[], DiarizationPipeline]
WaveformLoader = Callable[[Path], object]


class PyannoteSpeakerDiarizer:
    """Run an overlap-aware exclusive diarization pipeline off the event loop."""

    def __init__(
        self,
        settings: Settings,
        store: LocalEphemeralBlobStore,
        *,
        pipeline_factory: PipelineFactory | None = None,
        waveform_loader: WaveformLoader | None = None,
        observability: Observability | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._pipeline_factory = pipeline_factory
        self._waveform_loader = waveform_loader or self._waveform_payload
        self._observability = observability or NoopObservability()
        self._pipeline: DiarizationPipeline | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def diarize(
        self,
        audio: BlobReference,
        *,
        expected_speakers: int | None = None,
    ) -> tuple[DiarizationTurn, ...]:
        """Return exclusive turns while preserving detected overlap flags."""

        path = self._store.resolve_path(audio)
        pipeline = await self._get_pipeline()
        kwargs: dict[str, object] = {}
        if expected_speakers is not None:
            kwargs["num_speakers"] = expected_speakers
        with self._observability.span(
            "speech.inference.diarization",
            kind="internal",
            attributes={
                "speech.inference.operation": "diarization",
                "speech.inference.device": self._settings.diarization_device,
            },
        ) as span:
            try:
                async with self._inference_lock:
                    started_at = time.perf_counter()
                    outcome = "failed"
                    self._observability.inference_started(
                        operation="diarization",
                        device=self._settings.diarization_device,
                    )
                    try:
                        output = await anyio.to_thread.run_sync(
                            lambda: pipeline(self._waveform_loader(path), **kwargs)
                        )
                        outcome = "completed"
                    finally:
                        span.set_attribute("speech.inference.outcome", outcome)
                        self._observability.inference_finished(
                            operation="diarization",
                            device=self._settings.diarization_device,
                            outcome=outcome,
                            duration_seconds=time.perf_counter() - started_at,
                        )
                return self._turns(output)
            except DiarizationUnavailableError:
                span.mark_error()
                raise
            except Exception:
                span.mark_error()
                raise DiarizationUnavailableError from None

    @staticmethod
    def _waveform_payload(path: Path) -> dict[str, object]:
        """Load normalized PCM16 WAV without relying on TorchCodec file decoding."""

        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if (
                channels < 1
                or sample_rate < 1
                or frame_count < 1
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
            ):
                raise DiarizationUnavailableError
            pcm = bytearray(audio.readframes(frame_count))

        import torch

        samples = torch.frombuffer(pcm, dtype=torch.int16)
        if samples.numel() == 0 or samples.numel() % channels:
            raise DiarizationUnavailableError
        waveform = (
            samples.reshape(-1, channels)
            .transpose(0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(32768.0)
        )
        return {"waveform": waveform, "sample_rate": sample_rate}

    async def _get_pipeline(self) -> DiarizationPipeline:
        if self._pipeline is not None:
            return self._pipeline
        async with self._load_lock:
            if self._pipeline is None:
                try:
                    self._pipeline = await anyio.to_thread.run_sync(self._load_pipeline)
                except DiarizationUnavailableError:
                    raise
                except Exception:
                    raise DiarizationUnavailableError from None
            return self._pipeline

    def _load_pipeline(self) -> DiarizationPipeline:
        if self._pipeline_factory is not None:
            return self._pipeline_factory()
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        source = self._settings.diarization_model_source
        if self._settings.diarization_model_local_files_only and not Path(source).is_dir():
            raise DiarizationUnavailableError
        try:
            from pyannote.audio import Pipeline

            token = self._settings.diarization_huggingface_token
            pipeline = Pipeline.from_pretrained(
                source,
                token=token.get_secret_value() if token is not None else None,
            )
            if pipeline is None:
                raise DiarizationUnavailableError
            if self._settings.diarization_device == "cuda":
                import torch

                pipeline.to(torch.device("cuda"))
            return cast(DiarizationPipeline, pipeline)
        except DiarizationUnavailableError:
            raise
        except Exception:
            raise DiarizationUnavailableError from None

    @classmethod
    def _turns(cls, output: object) -> tuple[DiarizationTurn, ...]:
        regular = getattr(output, "speaker_diarization", None)
        exclusive = getattr(output, "exclusive_speaker_diarization", regular)
        if regular is None or exclusive is None:
            raise DiarizationUnavailableError
        regular_entries = cls._entries(regular)
        exclusive_entries = cls._entries(exclusive)
        overlaps = cls._overlap_intervals(regular_entries)
        turns = tuple(
            DiarizationTurn(
                speaker=speaker,
                start_seconds=start,
                end_seconds=end,
                overlapping_speech=any(
                    min(end, overlap_end) - max(start, overlap_start) > 0
                    for overlap_start, overlap_end in overlaps
                ),
            )
            for start, end, speaker in exclusive_entries
            if end > start
        )
        if not turns:
            raise DiarizationUnavailableError
        return tuple(sorted(turns, key=lambda turn: (turn.start_seconds, turn.end_seconds)))

    @classmethod
    def _entries(cls, annotation: object) -> tuple[tuple[float, float, str], ...]:
        itertracks = getattr(annotation, "itertracks", None)
        values: Iterable[Any] = itertracks(yield_label=True) if callable(itertracks) else annotation  # type: ignore[assignment]
        entries: list[tuple[float, float, str]] = []
        for value in values:
            if not isinstance(value, tuple) or len(value) not in {2, 3}:
                raise DiarizationUnavailableError
            segment = value[0]
            speaker = value[-1]
            start = getattr(segment, "start", None)
            end = getattr(segment, "end", None)
            if not isinstance(start, int | float) or not isinstance(end, int | float):
                raise DiarizationUnavailableError
            entries.append((float(start), float(end), str(speaker)))
        return tuple(entries)

    @staticmethod
    def _overlap_intervals(
        entries: tuple[tuple[float, float, str], ...],
    ) -> tuple[tuple[float, float], ...]:
        overlaps: list[tuple[float, float]] = []
        for index, (start, end, speaker) in enumerate(entries):
            for other_start, other_end, other_speaker in entries[index + 1 :]:
                if speaker == other_speaker:
                    continue
                overlap_start = max(start, other_start)
                overlap_end = min(end, other_end)
                if overlap_end > overlap_start:
                    overlaps.append((overlap_start, overlap_end))
        return tuple(overlaps)
