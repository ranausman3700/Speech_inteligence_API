"""Temporary PCM snapshot boundary for live speech recognition."""

from datetime import datetime
from typing import Protocol

from speech_intelligence_api.domain.models import BlobReference


class PcmAudioSnapshotStore(Protocol):
    """Create and immediately remove private WAV snapshots from bounded PCM."""

    async def create(
        self,
        pcm_audio: bytes,
        *,
        sample_rate_hz: int,
        expires_at: datetime,
    ) -> BlobReference:
        """Persist one mono signed-16-bit PCM snapshot as an expiring WAV."""

    async def delete(self, reference: BlobReference) -> bool:
        """Permanently remove a completed or failed inference snapshot."""
