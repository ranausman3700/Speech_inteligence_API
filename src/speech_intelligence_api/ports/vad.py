"""Voice-activity-detection boundary."""

from typing import Protocol

from speech_intelligence_api.domain.models import SpeechRegion


class VoiceActivityDetector(Protocol):
    """Detect speech without coupling sessions to a VAD implementation."""

    async def detect(self, pcm_audio: bytes, *, sample_rate_hz: int) -> tuple[SpeechRegion, ...]:
        """Return speech regions for mono signed 16-bit PCM audio."""
