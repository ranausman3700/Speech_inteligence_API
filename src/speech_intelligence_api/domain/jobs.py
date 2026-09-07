"""Immutable asynchronous-job payload and result objects."""

from __future__ import annotations

from dataclasses import dataclass

from speech_intelligence_api.domain.enums import JobQueue
from speech_intelligence_api.domain.models import (
    ConversationResult,
    JobRecord,
    TranscriptionRequest,
    TranscriptionResult,
)


@dataclass(frozen=True, slots=True)
class TranscriptionJobPayload:
    """Private worker input retained only for the job lifetime."""

    request: TranscriptionRequest
    duration_seconds: float
    queue: JobQueue

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("job audio duration must be positive")
        if self.queue not in {JobQueue.SHORT_TRANSCRIPTION, JobQueue.LONG_AUDIO}:
            raise ValueError("transcription jobs require a transcription queue")


@dataclass(frozen=True, slots=True)
class TranscriptionJobResult:
    """Completed transcription retained for the remaining privacy window."""

    result: TranscriptionResult
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("result audio duration must be positive")


@dataclass(frozen=True, slots=True)
class ConversationJobPayload:
    """Private diarization-worker input retained only for the job lifetime."""

    request: TranscriptionRequest
    duration_seconds: float
    expected_speakers: int | None
    speaker_confidence_threshold: float
    queue: JobQueue = JobQueue.DIARIZATION

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("conversation audio duration must be positive")
        if self.expected_speakers is not None and not 2 <= self.expected_speakers <= 100:
            raise ValueError("expected speakers must be between 2 and 100")
        if not 0 <= self.speaker_confidence_threshold <= 1:
            raise ValueError("speaker confidence threshold must be between 0 and 1")
        if self.queue is not JobQueue.DIARIZATION:
            raise ValueError("conversation jobs require the diarization queue")


@dataclass(frozen=True, slots=True)
class ConversationJobResult:
    """Completed conversation result retained inside the privacy window."""

    result: ConversationResult
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("conversation result audio duration must be positive")


JobPayload = TranscriptionJobPayload | ConversationJobPayload
JobResult = TranscriptionJobResult | ConversationJobResult


@dataclass(frozen=True, slots=True)
class JobReservation:
    """Outcome of an atomic capacity and idempotency reservation."""

    record: JobRecord
    created: bool
