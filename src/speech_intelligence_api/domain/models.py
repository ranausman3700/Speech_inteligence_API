"""Immutable domain value objects shared by use cases and adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from speech_intelligence_api.domain.enums import (
    ChineseScript,
    JobKind,
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
    LiveEventType,
)

_MAX_PRIVATE_RETENTION = timedelta(minutes=30)


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class BlobReference:
    """Opaque reference to a private, expiring storage object."""

    key: str
    media_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.key or self.key.startswith(("/", "\\")) or ".." in self.key.split("/"):
            raise ValueError("blob key must be a non-empty relative opaque path")
        if self.size_bytes < 0:
            raise ValueError("blob size cannot be negative")
        _require_utc(self.created_at, "created_at")
        _require_utc(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("blob expiry must be after creation")
        if self.expires_at - self.created_at > _MAX_PRIVATE_RETENTION:
            raise ValueError("blob lifetime cannot exceed 30 minutes")


@dataclass(frozen=True, slots=True)
class AudioProbe:
    """Validated metadata read from the media container rather than client headers."""

    detected_media_type: str
    container_format: str
    codec: str
    channels: int
    sample_rate_hz: int
    duration_seconds: float | None

    def __post_init__(self) -> None:
        if not self.detected_media_type.startswith("audio/"):
            raise ValueError("detected media type must be audio")
        if not self.container_format or not self.codec:
            raise ValueError("container format and codec are required")
        if self.channels < 1 or self.sample_rate_hz < 1:
            raise ValueError("audio channels and sample rate must be positive")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("known audio duration must be positive")


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """Raw and normalized private artifacts plus authoritative duration metadata."""

    raw: BlobReference
    normalized: BlobReference
    probe: AudioProbe
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("prepared audio duration must be positive")


@dataclass(frozen=True, slots=True)
class LanguageSelection:
    """Validated explicit or automatic language-selection policy."""

    mode: LanguageSelectionMode
    language: LanguageCode | None = None
    chinese_script: ChineseScript | None = None

    def __post_init__(self) -> None:
        if self.mode is LanguageSelectionMode.EXPLICIT and self.language is None:
            raise ValueError("explicit language mode requires a language code")
        if self.mode is LanguageSelectionMode.AUTOMATIC and self.language is not None:
            raise ValueError("automatic language mode cannot force a language code")
        if self.language is LanguageCode.CHINESE and self.chinese_script is None:
            raise ValueError("explicit Chinese transcription requires a script")
        if self.language not in {None, LanguageCode.CHINESE} and self.chinese_script is not None:
            raise ValueError("Chinese script can only be set for Chinese or automatic mode")


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """Provider-neutral transcription input."""

    audio: BlobReference
    language: LanguageSelection
    vocabulary: tuple[str, ...] = ()
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        if len(self.vocabulary) > 100:
            raise ValueError("custom vocabulary cannot exceed 100 entries")
        if any(not item.strip() or len(item) > 100 for item in self.vocabulary):
            raise ValueError("custom vocabulary entries must contain 1 to 100 characters")


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """A word with monotonic timing and an optional model-derived score."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence_estimate: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("word text cannot be empty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("word timestamps must be non-negative and monotonic")
        if self.confidence_estimate is not None and not 0 <= self.confidence_estimate <= 1:
            raise ValueError("word confidence estimate must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A transcript segment with optional speaker and overlap metadata."""

    text: str
    start_seconds: float
    end_seconds: float
    language: LanguageCode
    words: tuple[TranscriptWord, ...] = ()
    speaker: str | None = None
    speaker_uncertain: bool = False
    speaker_confidence_estimate: float | None = None
    overlapping_speech: bool = False
    confidence_estimate: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("segment text cannot be empty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("segment timestamps must be non-negative and monotonic")
        if self.confidence_estimate is not None and not 0 <= self.confidence_estimate <= 1:
            raise ValueError("segment confidence estimate must be between 0 and 1")
        if (
            self.speaker_confidence_estimate is not None
            and not 0 <= self.speaker_confidence_estimate <= 1
        ):
            raise ValueError("speaker confidence estimate must be between 0 and 1")
        if self.speaker is None and (
            self.speaker_uncertain or self.speaker_confidence_estimate is not None
        ):
            raise ValueError("speaker metadata requires a speaker label")
        if any(
            word.start_seconds < self.start_seconds or word.end_seconds > self.end_seconds
            for word in self.words
        ):
            raise ValueError("word timestamps must fall within their segment")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Native-script transcription output."""

    language: LanguageCode
    language_confidence_estimate: float
    text: str
    segments: tuple[TranscriptSegment, ...]
    chinese_script: ChineseScript | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.language_confidence_estimate <= 1:
            raise ValueError("language confidence estimate must be between 0 and 1")
        if self.language is LanguageCode.CHINESE and self.chinese_script is None:
            raise ValueError("Chinese results must identify their output script")
        if self.language is not LanguageCode.CHINESE and self.chinese_script is not None:
            raise ValueError("non-Chinese results cannot identify a Chinese output script")


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """A VAD-produced speech interval."""

    start_seconds: float
    end_seconds: float
    confidence_estimate: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("speech-region timestamps must be positive and monotonic")
        if not 0 <= self.confidence_estimate <= 1:
            raise ValueError("VAD confidence estimate must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class LiveTranscriptionOptions:
    """Validated options fixed for the lifetime of a live PCM session."""

    language: LanguageSelection
    sample_rate_hz: int = 16_000
    vocabulary: tuple[str, ...] = ()
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("live transcription requires 16 kHz audio")
        if len(self.vocabulary) > 100:
            raise ValueError("custom vocabulary cannot exceed 100 entries")
        if any(not item.strip() or len(item) > 100 for item in self.vocabulary):
            raise ValueError("custom vocabulary entries must contain 1 to 100 characters")


@dataclass(frozen=True, slots=True)
class LiveSessionEvent:
    """Transport-neutral speech boundary or native-script transcript update."""

    event_type: LiveEventType
    utterance_id: int
    revision: int
    start_seconds: float
    end_seconds: float
    result: TranscriptionResult | None = None

    def __post_init__(self) -> None:
        if self.utterance_id < 1:
            raise ValueError("utterance ID must be positive")
        if self.revision < 0:
            raise ValueError("live transcript revision cannot be negative")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("live event timestamps must be non-negative and monotonic")
        has_transcript = self.event_type in {LiveEventType.PARTIAL, LiveEventType.FINAL}
        if has_transcript is not (self.result is not None):
            raise ValueError("only transcript events may contain a transcription result")
        if self.event_type is LiveEventType.SPEECH_STARTED and self.revision != 0:
            raise ValueError("speech-started events must use revision zero")


@dataclass(frozen=True, slots=True)
class DiarizationTurn:
    """A speaker turn independent of transcript text."""

    speaker: str
    start_seconds: float
    end_seconds: float
    confidence_estimate: float | None = None
    overlapping_speech: bool = False

    def __post_init__(self) -> None:
        if not self.speaker:
            raise ValueError("speaker label cannot be empty")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("speaker-turn timestamps must be positive and monotonic")
        if self.confidence_estimate is not None and not 0 <= self.confidence_estimate <= 1:
            raise ValueError("speaker confidence estimate must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ConversationResult:
    """Combined diarization and transcription result."""

    language: LanguageCode
    language_confidence_estimate: float
    chinese_script: ChineseScript | None
    raw_transcript: str
    formatted_transcript: str
    segments: tuple[TranscriptSegment, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.language_confidence_estimate <= 1:
            raise ValueError("language confidence estimate must be between 0 and 1")
        if self.language is LanguageCode.CHINESE and self.chinese_script is None:
            raise ValueError("Chinese conversations must identify their output script")
        if self.language is not LanguageCode.CHINESE and self.chinese_script is not None:
            raise ValueError("non-Chinese conversations cannot identify a Chinese output script")
        if not self.raw_transcript or not self.formatted_transcript:
            raise ValueError("conversation transcripts cannot be empty")
        if any(segment.speaker is None for segment in self.segments):
            raise ValueError("conversation segments require speaker labels")


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Ephemeral asynchronous job state."""

    job_id: str
    kind: JobKind
    status: JobStatus
    created_at: datetime
    expires_at: datetime
    progress_percent: int = 0
    cancellation_requested: bool = False
    task_id: str | None = None
    failure_code: str | None = None
    version: int = 0

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job ID cannot be empty")
        _require_utc(self.created_at, "created_at")
        _require_utc(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("job expiry must be after creation")
        if self.expires_at - self.created_at > _MAX_PRIVATE_RETENTION:
            raise ValueError("job lifetime cannot exceed 30 minutes")
        if not 0 <= self.progress_percent <= 100:
            raise ValueError("job progress must be between 0 and 100")
        if self.task_id is not None and not self.task_id:
            raise ValueError("task ID cannot be empty")
        if self.failure_code is not None and not self.failure_code:
            raise ValueError("failure code cannot be empty")
        if self.version < 0:
            raise ValueError("job version cannot be negative")

    def transition(
        self,
        target: JobStatus,
        *,
        progress_percent: int | None = None,
        failure_code: str | None = None,
    ) -> JobRecord:
        """Return a new record after a legal state transition."""

        if not self.status.can_transition_to(target):
            raise ValueError(f"illegal job transition: {self.status.value} -> {target.value}")
        progress = self.progress_percent if progress_percent is None else progress_percent
        if progress < self.progress_percent:
            raise ValueError("job progress cannot move backwards")
        if target is JobStatus.SUCCEEDED:
            progress = 100
        return replace(
            self,
            status=target,
            progress_percent=progress,
            failure_code=failure_code,
            version=self.version + 1,
        )

    def with_task_id(self, task_id: str) -> JobRecord:
        """Associate a broker task without changing lifecycle state."""

        if not task_id:
            raise ValueError("task ID cannot be empty")
        return replace(self, task_id=task_id, version=self.version + 1)
