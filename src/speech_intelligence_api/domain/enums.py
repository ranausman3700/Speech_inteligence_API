"""Domain enumerations and the supported-language allowlist."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LanguageCode(StrEnum):
    """Language codes accepted by Faster-Whisper for this service."""

    ENGLISH = "en"
    SPANISH = "es"
    PORTUGUESE = "pt"
    ARABIC = "ar"
    FRENCH = "fr"
    GERMAN = "de"
    RUSSIAN = "ru"
    TURKISH = "tr"
    ITALIAN = "it"
    KOREAN = "ko"
    JAPANESE = "ja"
    CHINESE = "zh"
    HINDI = "hi"
    INDONESIAN = "id"
    THAI = "th"
    VIETNAMESE = "vi"
    MALAY = "ms"
    DUTCH = "nl"
    POLISH = "pl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    HEBREW = "he"
    UKRAINIAN = "uk"


class ChineseScript(StrEnum):
    """Supported Chinese output scripts."""

    SIMPLIFIED = "simplified"
    TRADITIONAL = "traditional"


class LanguageSelectionMode(StrEnum):
    """How the transcription language is selected."""

    EXPLICIT = "explicit"
    AUTOMATIC = "automatic"


class ProcessingMode(StrEnum):
    """Whether a request may execute directly or must use a background job."""

    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


class LiveEventType(StrEnum):
    """Server events emitted during one live dictation session."""

    SPEECH_STARTED = "speech_started"
    PARTIAL = "partial"
    FINAL = "final"


class ExportContent(StrEnum):
    """What a caller wants written into an exported document."""

    TRANSCRIPT = "transcript"
    SUMMARY = "summary"


class ExportFormat(StrEnum):
    """Supported export document formats."""

    TXT = "txt"
    PDF = "pdf"


class JobKind(StrEnum):
    """Asynchronous workload categories."""

    TRANSCRIPTION = "transcription"
    CONVERSATION = "conversation"


class JobQueue(StrEnum):
    """Named worker queues used to isolate workload classes."""

    SHORT_TRANSCRIPTION = "transcription.short"
    LONG_AUDIO = "transcription.long"
    DIARIZATION = "diarization"
    CLEANUP = "cleanup"


class JobStatus(StrEnum):
    """Externally visible job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        """Return whether this state permits no further transitions."""

        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.EXPIRED,
        }

    def can_transition_to(self, target: JobStatus) -> bool:
        """Enforce the legal job state machine."""

        transitions = {
            JobStatus.QUEUED: {
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
                JobStatus.FAILED,
                JobStatus.EXPIRED,
            },
            JobStatus.RUNNING: {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLING,
                JobStatus.CANCELLED,
                JobStatus.EXPIRED,
            },
            JobStatus.CANCELLING: {
                JobStatus.CANCELLED,
                JobStatus.FAILED,
                JobStatus.EXPIRED,
            },
            JobStatus.SUCCEEDED: {JobStatus.EXPIRED},
            JobStatus.FAILED: {JobStatus.EXPIRED},
            JobStatus.CANCELLED: {JobStatus.EXPIRED},
            JobStatus.EXPIRED: set(),
        }
        return target in transitions[self]


@dataclass(frozen=True, slots=True)
class LanguageVariant:
    """A user-facing supported language/script combination."""

    name: str
    code: LanguageCode
    chinese_script: ChineseScript | None = None


SUPPORTED_LANGUAGE_VARIANTS: tuple[LanguageVariant, ...] = (
    LanguageVariant("English", LanguageCode.ENGLISH),
    LanguageVariant("Spanish", LanguageCode.SPANISH),
    LanguageVariant("Portuguese", LanguageCode.PORTUGUESE),
    LanguageVariant("Arabic", LanguageCode.ARABIC),
    LanguageVariant("French", LanguageCode.FRENCH),
    LanguageVariant("German", LanguageCode.GERMAN),
    LanguageVariant("Russian", LanguageCode.RUSSIAN),
    LanguageVariant("Turkish", LanguageCode.TURKISH),
    LanguageVariant("Italian", LanguageCode.ITALIAN),
    LanguageVariant("Korean", LanguageCode.KOREAN),
    LanguageVariant("Japanese", LanguageCode.JAPANESE),
    LanguageVariant("Chinese Simplified", LanguageCode.CHINESE, ChineseScript.SIMPLIFIED),
    LanguageVariant("Chinese Traditional", LanguageCode.CHINESE, ChineseScript.TRADITIONAL),
    LanguageVariant("Hindi", LanguageCode.HINDI),
    LanguageVariant("Indonesian", LanguageCode.INDONESIAN),
    LanguageVariant("Thai", LanguageCode.THAI),
    LanguageVariant("Vietnamese", LanguageCode.VIETNAMESE),
    LanguageVariant("Malay", LanguageCode.MALAY),
    LanguageVariant("Dutch", LanguageCode.DUTCH),
    LanguageVariant("Polish", LanguageCode.POLISH),
    LanguageVariant("Swedish", LanguageCode.SWEDISH),
    LanguageVariant("Norwegian", LanguageCode.NORWEGIAN),
    LanguageVariant("Danish", LanguageCode.DANISH),
    LanguageVariant("Hebrew", LanguageCode.HEBREW),
    LanguageVariant("Ukrainian", LanguageCode.UKRAINIAN),
)
