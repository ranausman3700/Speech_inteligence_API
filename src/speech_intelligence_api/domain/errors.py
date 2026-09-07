"""Public-safe service error taxonomy."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable machine-readable API error codes."""

    AUTHENTICATION_FAILED = "authentication_failed"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    INVALID_AUDIO = "invalid_audio"
    AUDIO_TOO_LONG = "audio_too_long"
    LANGUAGE_UNCERTAIN = "language_uncertain"
    ASYNC_PROCESSING_REQUIRED = "async_processing_required"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    INTERNAL_ERROR = "internal_error"


class ServiceError(Exception):
    """An expected error containing only explicitly public response data."""

    def __init__(
        self,
        code: ErrorCode,
        public_message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.public_message = public_message
        self.details = dict(details or {})


class AuthenticationError(ServiceError):
    """Raised when an API key is absent or invalid."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.AUTHENTICATION_FAILED,
            "A valid API key is required.",
        )


class PayloadTooLargeError(ServiceError):
    """Raised before an upload can exceed its configured byte limit."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__(
            ErrorCode.PAYLOAD_TOO_LARGE,
            "The uploaded file exceeds the configured size limit.",
            details={"max_bytes": max_bytes},
        )


class UnsupportedMediaTypeError(ServiceError):
    """Raised when declared, named, or detected media types are not accepted."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            "The uploaded file is not a supported audio type.",
        )


class InvalidAudioError(ServiceError):
    """Raised when media cannot be safely decoded into speech audio."""

    def __init__(self, message: str = "The uploaded file is not valid decodable audio.") -> None:
        super().__init__(ErrorCode.INVALID_AUDIO, message)


class AudioTooLongError(ServiceError):
    """Raised when decoded audio exceeds its privacy/resource boundary."""

    def __init__(self, max_duration_seconds: float) -> None:
        super().__init__(
            ErrorCode.AUDIO_TOO_LONG,
            "The audio duration exceeds the configured limit.",
            details={"max_duration_seconds": max_duration_seconds},
        )


class UncertainLanguageError(ServiceError):
    """Raised before decoding text when automatic language confidence is insufficient."""

    def __init__(
        self,
        threshold: float,
        candidates: tuple[tuple[str, float], ...],
    ) -> None:
        super().__init__(
            ErrorCode.LANGUAGE_UNCERTAIN,
            "The speech language could not be identified with sufficient confidence.",
            details={
                "confidence_threshold": threshold,
                "candidates": [
                    {"language": language, "confidence_estimate": confidence}
                    for language, confidence in candidates
                ],
            },
        )


class AsyncProcessingRequiredError(ServiceError):
    """Raised until Phase 3 can enqueue work that should not run inline."""

    def __init__(self, sync_max_duration_seconds: float) -> None:
        super().__init__(
            ErrorCode.ASYNC_PROCESSING_REQUIRED,
            "This recording requires asynchronous processing.",
            details={"sync_max_duration_seconds": sync_max_duration_seconds},
        )


class ModelUnavailableError(ServiceError):
    """Raised when the configured speech-recognition model cannot serve work."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.MODEL_UNAVAILABLE,
            "The speech-recognition model is temporarily unavailable.",
        )


class DiarizationUnavailableError(ServiceError):
    """Raised when the configured speaker model cannot serve work."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.MODEL_UNAVAILABLE,
            "The speaker-diarization model is temporarily unavailable.",
        )


class CapacityExceededError(ServiceError):
    """Raised when accepting more work would exceed configured backpressure."""

    def __init__(self, max_pending_jobs: int, *, retry_after_seconds: int = 5) -> None:
        super().__init__(
            ErrorCode.CAPACITY_EXCEEDED,
            "The asynchronous processing queue is currently full.",
            details={
                "max_pending_jobs": max_pending_jobs,
                "retry_after_seconds": retry_after_seconds,
            },
        )
        self.retry_after_seconds = retry_after_seconds


class RateLimitExceededError(ServiceError):
    """Raised when one caller exhausts an admission window."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        retry_after_seconds: int,
    ) -> None:
        super().__init__(
            ErrorCode.RATE_LIMITED,
            "The request rate limit has been exceeded.",
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after_seconds,
            },
        )
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds


class LiveCapacityExceededError(ServiceError):
    """Raised when one API process cannot admit another live session."""

    def __init__(self, max_sessions: int) -> None:
        super().__init__(
            ErrorCode.CAPACITY_EXCEEDED,
            "The live-transcription service is currently at capacity.",
            details={"max_sessions": max_sessions},
        )


class LiveSessionLimitError(ServiceError):
    """Raised when streamed audio crosses a bounded session limit."""

    def __init__(self, max_session_seconds: float) -> None:
        super().__init__(
            ErrorCode.INVALID_REQUEST,
            "The live-transcription session reached its audio-duration limit.",
            details={"max_session_seconds": max_session_seconds},
        )


class IdempotencyConflictError(ServiceError):
    """Raised when one idempotency key is reused for a different request."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.CONFLICT,
            "The idempotency key was already used for a different request.",
        )


class JobNotFoundError(ServiceError):
    """Raised when an ephemeral job does not exist or has expired."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.NOT_FOUND,
            "The job does not exist or has expired.",
        )


class JobNotReadyError(ServiceError):
    """Raised when a result is requested before successful completion."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.CONFLICT,
            "The job result is not available.",
        )


class DependencyUnavailableError(ServiceError):
    """Raised when durable queue state cannot be reached."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "The asynchronous processing service is temporarily unavailable.",
        )
