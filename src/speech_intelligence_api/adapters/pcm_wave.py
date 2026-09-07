"""Private PCM-to-WAV snapshots for bounded live inference."""

from __future__ import annotations

import wave
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import anyio

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.domain.models import BlobReference


class PcmWaveSnapshotStore:
    """Encode in-memory mono signed-16-bit PCM into private temporary WAV files."""

    def __init__(self, store: LocalEphemeralBlobStore) -> None:
        self._store = store

    async def create(
        self,
        pcm_audio: bytes,
        *,
        sample_rate_hz: int,
        expires_at: datetime,
    ) -> BlobReference:
        """Create an expiring WAV and remove partial output if encoding fails."""

        if not pcm_audio or len(pcm_audio) % 2:
            raise ValueError("PCM audio must contain whole signed 16-bit samples")
        if sample_rate_hz != 16_000:
            raise ValueError("live PCM snapshots require a 16 kHz sample rate")
        destination = self._store.reserve_path(".wav")
        try:
            await anyio.to_thread.run_sync(
                self._write_wave,
                destination,
                pcm_audio,
                sample_rate_hz,
            )
            return self._store.reference_for_path(
                destination,
                media_type="audio/wav",
                expires_at=expires_at,
            )
        except BaseException:
            with suppress(FileNotFoundError):
                destination.unlink()
            raise

    async def delete(self, reference: BlobReference) -> bool:
        """Delete one inference snapshot immediately after use."""

        return await self._store.delete(reference)

    @staticmethod
    def _write_wave(destination: Path, pcm_audio: bytes, sample_rate_hz: int) -> None:
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate_hz)
            output.writeframes(pcm_audio)
