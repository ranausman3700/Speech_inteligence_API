"""Audio probing and normalization boundary."""

from datetime import datetime
from typing import Protocol

from speech_intelligence_api.domain.models import AudioProbe, BlobReference


class AudioPreprocessor(Protocol):
    """Inspect and normalize private uploaded audio."""

    async def probe(self, source: BlobReference) -> AudioProbe:
        """Read authoritative container and codec metadata."""

    async def normalize(
        self,
        source: BlobReference,
        *,
        expires_at: datetime,
        max_duration_seconds: float,
    ) -> tuple[BlobReference, float]:
        """Create 16 kHz mono signed-16-bit WAV and return its actual duration."""
