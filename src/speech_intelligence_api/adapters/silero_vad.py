"""Silero ONNX voice-activity detection using Faster-Whisper's pinned asset."""

from __future__ import annotations

import math
from collections.abc import Callable

import anyio
import numpy as np
from numpy.typing import NDArray

from speech_intelligence_api.domain.models import SpeechRegion

ProbabilityModel = Callable[[NDArray[np.float32]], NDArray[np.float32]]

_WINDOW_SAMPLES = 512


class SileroVoiceActivityDetector:
    """Detect bounded speech regions without downloading models at runtime."""

    def __init__(
        self,
        *,
        threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
        probability_model: ProbabilityModel | None = None,
    ) -> None:
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._probability_model = probability_model

    async def detect(
        self,
        pcm_audio: bytes,
        *,
        sample_rate_hz: int,
    ) -> tuple[SpeechRegion, ...]:
        """Return speech regions for mono little-endian signed-16-bit PCM."""

        if not pcm_audio:
            return ()
        if len(pcm_audio) % 2:
            raise ValueError("PCM audio must contain whole signed 16-bit samples")
        if sample_rate_hz != 16_000:
            raise ValueError("Silero live VAD requires a 16 kHz sample rate")
        return await anyio.to_thread.run_sync(self._detect_sync, pcm_audio, sample_rate_hz)

    def _detect_sync(
        self,
        pcm_audio: bytes,
        sample_rate_hz: int,
    ) -> tuple[SpeechRegion, ...]:
        samples = np.frombuffer(pcm_audio, dtype="<i2").astype(np.float32)
        samples /= np.float32(32768.0)
        original_samples = int(samples.size)
        padding = (-original_samples) % _WINDOW_SAMPLES
        if padding:
            samples = np.pad(samples, (0, padding))

        probabilities = np.asarray(
            self._model()(samples),
            dtype=np.float32,
        ).reshape(-1)
        min_speech_samples = math.ceil(sample_rate_hz * self._min_speech_ms / 1000)
        min_silence_frames = max(
            1,
            math.ceil(sample_rate_hz * self._min_silence_ms / 1000 / _WINDOW_SAMPLES),
        )
        pad_samples = math.ceil(sample_rate_hz * self._speech_pad_ms / 1000)

        regions: list[SpeechRegion] = []
        start_frame: int | None = None
        last_voiced_frame: int | None = None
        for frame_index, probability in enumerate(probabilities):
            if float(probability) >= self._threshold:
                if start_frame is None:
                    start_frame = frame_index
                last_voiced_frame = frame_index
                continue
            if (
                start_frame is not None
                and last_voiced_frame is not None
                and frame_index - last_voiced_frame >= min_silence_frames
            ):
                self._append_region(
                    regions,
                    probabilities,
                    start_frame,
                    last_voiced_frame,
                    original_samples,
                    sample_rate_hz,
                    min_speech_samples,
                    pad_samples,
                )
                start_frame = None
                last_voiced_frame = None

        if start_frame is not None and last_voiced_frame is not None:
            self._append_region(
                regions,
                probabilities,
                start_frame,
                last_voiced_frame,
                original_samples,
                sample_rate_hz,
                min_speech_samples,
                pad_samples,
            )
        return tuple(regions)

    def _model(self) -> ProbabilityModel:
        if self._probability_model is None:
            from faster_whisper.vad import get_vad_model  # type: ignore[import-untyped]

            self._probability_model = get_vad_model()
        return self._probability_model

    @staticmethod
    def _append_region(
        regions: list[SpeechRegion],
        probabilities: NDArray[np.float32],
        start_frame: int,
        last_voiced_frame: int,
        original_samples: int,
        sample_rate_hz: int,
        min_speech_samples: int,
        pad_samples: int,
    ) -> None:
        voiced_start = start_frame * _WINDOW_SAMPLES
        voiced_end = min(original_samples, (last_voiced_frame + 1) * _WINDOW_SAMPLES)
        if voiced_end - voiced_start < min_speech_samples:
            return
        start_sample = max(0, voiced_start - pad_samples)
        end_sample = min(original_samples, voiced_end + pad_samples)
        confidence = float(
            np.mean(probabilities[start_frame : last_voiced_frame + 1], dtype=np.float64)
        )
        regions.append(
            SpeechRegion(
                start_seconds=start_sample / sample_rate_hz,
                end_seconds=end_sample / sample_rate_hz,
                confidence_estimate=min(1.0, max(0.0, confidence)),
            )
        )
