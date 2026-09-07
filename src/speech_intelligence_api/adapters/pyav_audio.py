"""PyAV-backed media probing and streaming audio normalization."""

from __future__ import annotations

import wave
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import av

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.domain.errors import AudioTooLongError, InvalidAudioError
from speech_intelligence_api.domain.models import AudioProbe, BlobReference

_TARGET_SAMPLE_RATE_HZ = 16_000
_SUPPORTED_CODECS = frozenset(
    {
        "aac",
        "alac",
        "flac",
        "mp3",
        "opus",
        "vorbis",
        "wavpack",
    }
)


class PyAvAudioPreprocessor:
    """Use bundled FFmpeg libraries through PyAV without shelling out."""

    def __init__(self, store: LocalEphemeralBlobStore) -> None:
        self._store = store

    async def probe(self, source: BlobReference) -> AudioProbe:
        """Read trusted metadata from the actual media container."""

        return await anyio.to_thread.run_sync(self._probe_sync, source)

    async def normalize(
        self,
        source: BlobReference,
        *,
        expires_at: datetime,
        max_duration_seconds: float,
    ) -> tuple[BlobReference, float]:
        """Stream-decode into a bounded mono PCM WAV object."""

        return await anyio.to_thread.run_sync(
            self._normalize_sync,
            source,
            expires_at,
            max_duration_seconds,
        )

    def _probe_sync(self, source: BlobReference) -> AudioProbe:
        try:
            with av.open(str(self._store.resolve_path(source)), mode="r") as container:
                stream = self._single_audio_stream(container)
                codec = str(stream.codec_context.name or "")
                self._validate_codec(codec)
                format_name = str(container.format.name or "")
                detected_media_type = self._media_type_for_format(format_name)
                duration_seconds = self._duration_seconds(container, stream)
                channels = int(stream.codec_context.channels or 0)
                sample_rate_hz = int(stream.codec_context.sample_rate or 0)
                return AudioProbe(
                    detected_media_type=detected_media_type,
                    container_format=format_name,
                    codec=codec,
                    channels=channels,
                    sample_rate_hz=sample_rate_hz,
                    duration_seconds=duration_seconds,
                )
        except InvalidAudioError:
            raise
        except Exception:
            raise InvalidAudioError from None

    def _normalize_sync(
        self,
        source: BlobReference,
        expires_at: datetime,
        max_duration_seconds: float,
    ) -> tuple[BlobReference, float]:
        destination = self._store.reserve_path(".wav")
        total_samples = 0
        max_samples = int(max_duration_seconds * _TARGET_SAMPLE_RATE_HZ)
        try:
            with (
                av.open(str(self._store.resolve_path(source)), mode="r") as container,
                wave.open(str(destination), "wb") as output,
            ):
                stream = self._single_audio_stream(container)
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(_TARGET_SAMPLE_RATE_HZ)
                resampler = av.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=_TARGET_SAMPLE_RATE_HZ,
                )
                for frame in container.decode(stream):
                    total_samples = self._write_resampled_frames(
                        output,
                        resampler.resample(frame),
                        total_samples,
                        max_samples,
                        max_duration_seconds,
                    )
                total_samples = self._write_resampled_frames(
                    output,
                    resampler.resample(None),
                    total_samples,
                    max_samples,
                    max_duration_seconds,
                )
                output.writeframes(b"")

            if total_samples == 0:
                raise InvalidAudioError("The uploaded audio contains no decodable samples.")
            duration_seconds = total_samples / _TARGET_SAMPLE_RATE_HZ
            reference = self._store.reference_for_path(
                destination,
                media_type="audio/wav",
                expires_at=expires_at,
            )
            return reference, duration_seconds
        except (AudioTooLongError, InvalidAudioError):
            self._delete_destination(destination)
            raise
        except Exception:
            self._delete_destination(destination)
            raise InvalidAudioError from None

    @staticmethod
    def _write_resampled_frames(
        output: wave.Wave_write,
        frames: list[av.AudioFrame],
        total_samples: int,
        max_samples: int,
        max_duration_seconds: float,
    ) -> int:
        for frame in frames:
            total_samples += frame.samples
            if total_samples > max_samples:
                raise AudioTooLongError(max_duration_seconds)
            byte_count = frame.samples * 2
            output.writeframesraw(bytes(frame.planes[0])[:byte_count])
        return total_samples

    @staticmethod
    def _single_audio_stream(container: Any) -> Any:
        audio_streams = container.streams.audio
        if len(audio_streams) != 1 or container.streams.video:
            raise InvalidAudioError("The upload must contain exactly one audio stream.")
        return audio_streams[0]

    @staticmethod
    def _validate_codec(codec: str) -> None:
        if not codec.startswith(("pcm_", "adpcm_")) and codec not in _SUPPORTED_CODECS:
            raise InvalidAudioError("The audio codec is not supported.")

    @staticmethod
    def _media_type_for_format(format_name: str) -> str:
        formats = set(format_name.casefold().split(","))
        if "wav" in formats:
            return "audio/wav"
        if "mp3" in formats:
            return "audio/mpeg"
        if "flac" in formats:
            return "audio/flac"
        if "ogg" in formats:
            return "audio/ogg"
        if formats.intersection({"matroska", "webm"}):
            return "audio/webm"
        if formats.intersection({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}):
            return "audio/mp4"
        if "aac" in formats:
            return "audio/aac"
        raise InvalidAudioError("The audio container is not supported.")

    @staticmethod
    def _duration_seconds(container: Any, stream: Any) -> float | None:
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        return None

    @staticmethod
    def _delete_destination(path: Path) -> None:
        with suppress(FileNotFoundError):
            path.unlink()
