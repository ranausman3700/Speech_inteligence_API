"""Native-script text normalization boundary."""

from typing import Protocol

from speech_intelligence_api.domain.models import TranscriptionResult


class TranscriptTextProcessor(Protocol):
    """Normalize safe Unicode and apply requested Chinese script conversion."""

    def process(self, result: TranscriptionResult) -> TranscriptionResult:
        """Return text without translation or romanization."""
