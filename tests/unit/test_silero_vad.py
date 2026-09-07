"""Silero VAD adapter tests, including its bundled ONNX runtime asset."""

from __future__ import annotations

import numpy as np
import pytest

from speech_intelligence_api.adapters.silero_vad import SileroVoiceActivityDetector


@pytest.mark.asyncio
async def test_silero_vad_groups_voiced_windows_with_confidence() -> None:
    def probabilities(_: np.ndarray) -> np.ndarray:
        return np.asarray([0.1, 0.8, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)

    detector = SileroVoiceActivityDetector(
        threshold=0.5,
        min_speech_ms=32,
        min_silence_ms=32,
        speech_pad_ms=0,
        probability_model=probabilities,
    )

    regions = await detector.detect(bytes(6 * 512 * 2), sample_rate_hz=16_000)

    assert len(regions) == 1
    assert regions[0].start_seconds == pytest.approx(0.032)
    assert regions[0].end_seconds == pytest.approx(0.096)
    assert regions[0].confidence_estimate == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_silero_vad_rejects_invalid_pcm_and_sample_rate() -> None:
    detector = SileroVoiceActivityDetector(
        threshold=0.5,
        min_speech_ms=32,
        min_silence_ms=32,
        speech_pad_ms=0,
        probability_model=lambda audio: np.zeros(audio.size // 512, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="whole signed 16-bit"):
        await detector.detect(b"x", sample_rate_hz=16_000)
    with pytest.raises(ValueError, match="16 kHz"):
        await detector.detect(b"\0\0", sample_rate_hz=8_000)
    assert await detector.detect(b"", sample_rate_hz=16_000) == ()


@pytest.mark.asyncio
async def test_silero_vad_ignores_speech_shorter_than_minimum() -> None:
    detector = SileroVoiceActivityDetector(
        threshold=0.5,
        min_speech_ms=64,
        min_silence_ms=32,
        speech_pad_ms=0,
        probability_model=lambda _: np.asarray([0.9, 0.1], dtype=np.float32),
    )

    assert await detector.detect(bytes(2 * 512 * 2), sample_rate_hz=16_000) == ()


@pytest.mark.asyncio
async def test_bundled_silero_model_processes_silence() -> None:
    detector = SileroVoiceActivityDetector(
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=160,
        speech_pad_ms=160,
    )

    assert await detector.detect(bytes(512 * 2), sample_rate_hz=16_000) == ()
