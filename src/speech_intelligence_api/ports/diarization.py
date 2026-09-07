"""Speaker-diarization boundary."""

from typing import Protocol

from speech_intelligence_api.domain.models import BlobReference, DiarizationTurn


class SpeakerDiarizer(Protocol):
    """Identify speaker turns in normalized audio."""

    async def diarize(
        self,
        audio: BlobReference,
        *,
        expected_speakers: int | None = None,
    ) -> tuple[DiarizationTurn, ...]:
        """Return speaker turns while retaining overlap and uncertainty information."""
