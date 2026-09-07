"""Small deterministic audio fixtures generated without external binaries."""

from __future__ import annotations

import io
import math
import struct
import wave


def make_wav_bytes(
    *,
    duration_seconds: float = 0.1,
    sample_rate_hz: int = 8_000,
    channels: int = 1,
    frequency_hz: float = 440.0,
) -> bytes:
    """Return a signed-16-bit PCM sine-wave WAV."""

    frame_count = int(duration_seconds * sample_rate_hz)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        for index in range(frame_count):
            sample = int(8_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz))
            output.writeframesraw(struct.pack("<h", sample) * channels)
        output.writeframes(b"")
    return buffer.getvalue()


def make_empty_wav_bytes(*, sample_rate_hz: int = 8_000) -> bytes:
    """Return a structurally valid WAV with no audio samples."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
    return buffer.getvalue()
