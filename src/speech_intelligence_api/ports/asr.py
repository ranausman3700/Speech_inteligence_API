"""Speech-recognition boundary."""

from typing import Protocol

from speech_intelligence_api.domain.models import TranscriptionRequest, TranscriptionResult


class SpeechRecognizer(Protocol):
    """Transcribe audio without exposing provider-specific types."""

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Return transcription in the source language and native script."""
