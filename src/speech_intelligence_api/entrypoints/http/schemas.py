"""Versioned HTTP response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from speech_intelligence_api.domain.enums import (
    ChineseScript,
    JobKind,
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
)


class ApiSchema(BaseModel):
    """Strict base for public API schemas."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiSchema):
    """Liveness or readiness response."""

    status: Literal["ok", "ready", "unavailable"]
    service: str
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class ValidationIssue(ApiSchema):
    """Sanitized request-validation issue."""

    location: str
    error_type: str


class ProblemDetail(ApiSchema):
    """RFC 9457-style sanitized API problem."""

    type: str
    title: str
    status: int
    detail: str
    code: str
    instance: str
    trace_id: str
    issues: list[ValidationIssue] | None = None
    details: dict[str, Any] | None = None


class LanguageCapability(ApiSchema):
    """Supported language/script combination."""

    name: str
    code: LanguageCode
    chinese_script: ChineseScript | None = None


class LiveTranscriptionCapability(ApiSchema):
    """Discoverable WebSocket audio and event contract."""

    websocket_path: str
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: Literal[16000] = 16_000
    channels: Literal[1] = 1
    partial_results: Literal[True] = True
    vad: Literal["silero"] = "silero"
    control_messages: tuple[Literal["commit", "stop", "ping"], ...] = (
        "commit",
        "stop",
        "ping",
    )
    server_events: tuple[
        Literal[
            "ready",
            "speech_started",
            "partial",
            "final",
            "pong",
            "error",
            "session_closed",
        ],
        ...,
    ] = (
        "ready",
        "speech_started",
        "partial",
        "final",
        "pong",
        "error",
        "session_closed",
    )


class ConversationCapability(ApiSchema):
    """Discoverable asynchronous speaker-diarization contract."""

    endpoint: str
    asynchronous_only: Literal[True] = True
    expected_speakers_minimum: int = 2
    expected_speakers_maximum: int
    word_timestamps: Literal[True] = True
    overlap_detection: Literal[True] = True


class CapabilitiesResponse(ApiSchema):
    """Stable capabilities contract exposed to API clients."""

    transcription_task: Literal["transcribe"] = "transcribe"
    translation_enabled: Literal[False] = False
    language_selection_modes: tuple[Literal["explicit", "automatic"], ...] = (
        "explicit",
        "automatic",
    )
    languages: list[LanguageCapability]
    live_transcription: LiveTranscriptionCapability | None = None
    conversations: ConversationCapability | None = None


class LiveStartMessage(ApiSchema):
    """First client message for an authenticated live session."""

    type: Literal["start"]
    api_key: SecretStr | None = Field(default=None, repr=False)
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: Literal[16000] = 16_000
    language_mode: LanguageSelectionMode = LanguageSelectionMode.AUTOMATIC
    language: LanguageCode | None = None
    chinese_script: ChineseScript | None = None
    vocabulary: list[str] = Field(default_factory=list, max_length=100)
    word_timestamps: bool = True


class LiveControlMessage(ApiSchema):
    """Client control message sent after the live handshake."""

    type: Literal["commit", "stop", "ping"]


class LiveReadyEventResponse(ApiSchema):
    """Accepted live session and immutable transport limits."""

    type: Literal["ready"] = "ready"
    request_id: str
    session_id: str
    expires_in_seconds: float
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: Literal[16000] = 16_000
    channels: Literal[1] = 1
    max_chunk_bytes: int


class LiveSpeechStartedEventResponse(ApiSchema):
    """Silero-detected start of one utterance."""

    type: Literal["speech_started"] = "speech_started"
    request_id: str
    session_id: str
    utterance_id: int
    start_seconds: float


class LivePongEventResponse(ApiSchema):
    """Application-level liveness acknowledgement."""

    type: Literal["pong"] = "pong"
    request_id: str
    session_id: str


class LiveErrorEventResponse(ApiSchema):
    """Sanitized WebSocket failure without private audio or credentials."""

    type: Literal["error"] = "error"
    request_id: str
    code: str
    detail: str
    fatal: bool
    details: dict[str, Any] | None = None


class LiveSessionClosedEventResponse(ApiSchema):
    """Orderly server acknowledgement after a client stop command."""

    type: Literal["session_closed"] = "session_closed"
    request_id: str
    session_id: str


class TranscriptWordResponse(ApiSchema):
    """Word timing and model-derived confidence estimate."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence_estimate: float | None


class TranscriptSegmentResponse(ApiSchema):
    """Timestamped native-script transcript segment."""

    text: str
    start_seconds: float
    end_seconds: float
    language: LanguageCode
    confidence_estimate: float | None
    words: list[TranscriptWordResponse]


class ConversationSegmentResponse(TranscriptSegmentResponse):
    """Timestamped transcript segment attributed to a conversation speaker."""

    speaker: str | None = None
    speaker_uncertain: bool = False
    speaker_confidence_estimate: float | None = None
    overlapping_speech: bool = False


class LiveTranscriptEventResponse(ApiSchema):
    """Best-effort partial or accurate finalized native-script transcript."""

    type: Literal["partial", "final"]
    request_id: str
    session_id: str
    utterance_id: int
    revision: int
    start_seconds: float
    end_seconds: float
    language: LanguageCode
    language_confidence_estimate: float
    chinese_script: ChineseScript | None
    text: str
    segments: list[TranscriptSegmentResponse]


class TranscriptionResponse(ApiSchema):
    """Completed direct batch transcription."""

    processing: Literal["completed"] = "completed"
    task: Literal["transcribe"] = "transcribe"
    request_id: str
    language: LanguageCode
    language_confidence_estimate: float
    chinese_script: ChineseScript | None
    duration_seconds: float
    text: str
    segments: list[TranscriptSegmentResponse]


class JobAcceptedResponse(ApiSchema):
    """Asynchronous job admission response."""

    processing: Literal["queued"] = "queued"
    task: Literal["transcribe", "diarize"] = "transcribe"
    request_id: str
    job_id: str
    kind: JobKind
    status: JobStatus
    progress_percent: int
    created_at: str
    expires_at: str
    replayed: bool
    links: dict[str, str]


class JobStatusResponse(ApiSchema):
    """Current state of an expiring background job."""

    request_id: str
    job_id: str
    kind: JobKind
    status: JobStatus
    progress_percent: int
    cancellation_requested: bool
    failure_code: str | None
    created_at: str
    expires_at: str
    links: dict[str, str]


class JobTranscriptionResultResponse(TranscriptionResponse):
    """Successful asynchronous transcription result."""

    job_id: str


class JobConversationResultResponse(ApiSchema):
    """Successful asynchronous native-script conversation result."""

    processing: Literal["completed"] = "completed"
    task: Literal["diarize"] = "diarize"
    request_id: str
    job_id: str
    language: LanguageCode
    language_confidence_estimate: float
    chinese_script: ChineseScript | None
    duration_seconds: float
    raw_transcript: str
    formatted_transcript: str
    segments: list[ConversationSegmentResponse]
