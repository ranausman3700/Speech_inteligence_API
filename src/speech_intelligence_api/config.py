"""Environment-backed application configuration."""

from __future__ import annotations

import re
import tempfile
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from speech_intelligence_api.domain.enums import ChineseScript

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HTTP_HEADER_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_TRUSTED_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_LOCAL_TRUSTED_HOSTS = ("localhost", "127.0.0.1", "testserver")


class Environment(StrEnum):
    """Supported deployment environments."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated service settings loaded from ``SPEECH_API_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="SPEECH_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    service_name: str = Field(default="speech-intelligence-api", min_length=1, max_length=80)
    service_version: str = Field(default="0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    api_prefix: str = Field(default="/v1", pattern=r"^/[a-z0-9/_-]*[a-z0-9]$")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    docs_enabled: bool = True
    auth_enabled: bool = False
    api_key_hmac_secret: SecretStr | None = Field(default=None, repr=False)
    api_key_digests: tuple[str, ...] = ()
    privacy_ttl_seconds: int = Field(default=1800, ge=60, le=1800)
    request_id_header: str = "X-Request-ID"
    trusted_hosts: tuple[str, ...] = _LOCAL_TRUSTED_HOSTS
    security_headers_enabled: bool = True
    hsts_max_age_seconds: int = Field(default=31_536_000, ge=0, le=63_072_000)
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, ge=1024, le=1024 * 1024 * 1024)
    max_audio_duration_seconds: float = Field(default=7200, ge=1, le=86_400)
    sync_max_audio_duration_seconds: float = Field(default=120, ge=1, le=3600)
    upload_chunk_bytes: int = Field(default=1024 * 1024, ge=4096, le=8 * 1024 * 1024)
    http_max_concurrency: int = Field(default=1200, ge=1000, le=100_000)
    http_backlog: int = Field(default=2048, ge=1000, le=65_535)
    http_graceful_shutdown_seconds: int = Field(default=30, ge=5, le=300)
    temp_storage_root: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "speech-intelligence-api"
    )
    asr_model_name: str = Field(default="large-v3", min_length=1, max_length=255)
    asr_device: Literal["auto", "cpu", "cuda"] = "auto"
    asr_compute_type: str = Field(default="default", min_length=1, max_length=32)
    asr_cpu_threads: int = Field(default=0, ge=0)
    asr_num_workers: int = Field(default=1, ge=1, le=32)
    asr_max_concurrency: int = Field(default=1, ge=1, le=32)
    asr_beam_size: int = Field(default=5, ge=1, le=20)
    asr_language_detection_segments: int = Field(default=3, ge=1, le=10)
    language_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    default_chinese_script: ChineseScript = ChineseScript.SIMPLIFIED
    asr_model_download_root: Path | None = None
    asr_model_local_files_only: bool = False
    async_jobs_enabled: bool = False
    redis_job_url: SecretStr = Field(
        default=SecretStr("redis://127.0.0.1:6379/1"),
        repr=False,
    )
    celery_broker_url: SecretStr = Field(
        default=SecretStr("redis://127.0.0.1:6379/0"),
        repr=False,
    )
    job_key_prefix: str = Field(
        default="speech-intelligence",
        pattern=r"^[a-z0-9][a-z0-9:_-]{1,62}[a-z0-9]$",
    )
    rate_limit_enabled: bool = False
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    rate_limit_http_requests: int = Field(default=120, ge=1, le=100_000)
    rate_limit_upload_requests: int = Field(default=20, ge=1, le=10_000)
    rate_limit_live_sessions: int = Field(default=10, ge=1, le=10_000)
    metrics_enabled: bool = False
    tracing_enabled: bool = False
    otlp_traces_endpoint: str = Field(
        default="http://127.0.0.1:4318/v1/traces",
        min_length=12,
        max_length=2048,
    )
    trace_sample_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    trace_export_timeout_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
    max_pending_jobs: int = Field(default=1000, ge=1, le=100_000)
    long_audio_queue_threshold_seconds: float = Field(default=600, ge=1, le=7200)
    job_soft_time_limit_seconds: int = Field(default=1500, ge=30, le=1740)
    job_hard_time_limit_seconds: int = Field(default=1740, ge=60, le=1770)
    job_max_retries: int = Field(default=3, ge=0, le=10)
    job_retry_backoff_seconds: int = Field(default=5, ge=1, le=300)
    celery_visibility_timeout_seconds: int = Field(default=1800, ge=60, le=86_400)
    celery_worker_prefetch_multiplier: int = Field(default=1, ge=1, le=16)
    artifact_cleanup_interval_seconds: int = Field(default=30, ge=30, le=1800)
    diarization_enabled: bool = False
    diarization_model_source: str = Field(
        default="pyannote/speaker-diarization-community-1",
        min_length=1,
        max_length=1024,
    )
    diarization_device: Literal["cpu", "cuda"] = "cpu"
    diarization_huggingface_token: SecretStr | None = Field(default=None, repr=False)
    diarization_model_local_files_only: bool = False
    diarization_max_expected_speakers: int = Field(default=20, ge=2, le=100)
    speaker_confidence_threshold: float = Field(default=0.60, ge=0, le=1)
    cors_allowed_origins: tuple[str, ...] = ()
    live_transcription_enabled: bool = True
    live_sample_rate_hz: int = Field(default=16_000, ge=16_000, le=16_000)
    live_max_sessions: int = Field(default=4, ge=1, le=256)
    live_max_session_seconds: float = Field(default=300, ge=10, le=1800)
    live_max_utterance_seconds: float = Field(default=30, ge=2, le=120)
    live_max_chunk_bytes: int = Field(default=32_000, ge=1024, le=256 * 1024)
    live_input_queue_chunks: int = Field(default=128, ge=4, le=1024)
    live_handshake_timeout_seconds: float = Field(default=10, ge=1, le=60)
    live_idle_timeout_seconds: float = Field(default=30, ge=5, le=300)
    live_vad_threshold: float = Field(default=0.5, ge=0.05, le=0.99)
    live_vad_min_speech_ms: int = Field(default=250, ge=32, le=2000)
    live_vad_min_silence_ms: int = Field(default=160, ge=32, le=2000)
    live_end_silence_ms: int = Field(default=700, ge=100, le=5000)
    live_speech_pad_ms: int = Field(default=160, ge=0, le=1000)
    live_vad_analysis_window_seconds: float = Field(default=3, ge=1, le=10)
    live_pre_roll_ms: int = Field(default=320, ge=0, le=2000)
    live_partial_min_audio_seconds: float = Field(default=1, ge=0.25, le=10)
    live_partial_interval_seconds: float = Field(default=2, ge=0.5, le=10)

    @field_validator("api_key_digests")
    @classmethod
    def validate_api_key_digests(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize digests and reject malformed or duplicate values."""

        normalized = tuple(value.strip().lower() for value in values)
        if any(not _SHA256_HEX_PATTERN.fullmatch(value) for value in normalized):
            raise ValueError("every API-key digest must be 64 lowercase hexadecimal characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("API-key digests must be unique")
        return normalized

    @field_validator("request_id_header")
    @classmethod
    def validate_request_id_header(cls, value: str) -> str:
        """Ensure the configured request-ID name is a legal HTTP field name."""

        if not _HTTP_HEADER_PATTERN.fullmatch(value):
            raise ValueError("request-ID header must be a valid HTTP field name")
        return value

    @field_validator("trusted_hosts")
    @classmethod
    def validate_trusted_hosts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require explicit, normalized hostnames without ports or wildcards."""

        normalized = tuple(value.strip().lower().rstrip(".") for value in values)
        if not normalized:
            raise ValueError("at least one trusted host is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("trusted hosts must be unique")
        if any(
            not value or "*" in value or ":" in value or not _TRUSTED_HOST_PATTERN.fullmatch(value)
            for value in normalized
        ):
            raise ValueError("trusted hosts must be explicit hostnames without ports")
        return normalized

    @field_validator("otlp_traces_endpoint")
    @classmethod
    def validate_otlp_traces_endpoint(cls, value: str) -> str:
        """Accept an explicit operator-controlled HTTP(S) OTLP trace endpoint."""

        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OTLP traces endpoint must be an explicit HTTP(S) URL")
        return value

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_allowed_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Accept only explicit HTTP origins without paths or wildcards."""

        from urllib.parse import urlsplit

        normalized = tuple(value.rstrip("/") for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("CORS origins must be unique")
        for value in normalized:
            parsed = urlsplit(value)
            if (
                value == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("CORS origins must be explicit HTTP origins")
        return normalized

    @model_validator(mode="after")
    def validate_security_posture(self) -> Self:
        """Fail closed in non-local environments and validate key material."""

        if (
            self.environment in {Environment.STAGING, Environment.PRODUCTION}
            and not self.auth_enabled
        ):
            raise ValueError(
                "API-key authentication is required outside local and test environments"
            )
        if (
            self.environment in {Environment.STAGING, Environment.PRODUCTION}
            and not self.rate_limit_enabled
        ):
            raise ValueError("rate limiting is required outside local and test environments")
        if self.environment in {Environment.STAGING, Environment.PRODUCTION}:
            if not self.security_headers_enabled:
                raise ValueError(
                    "security headers are required outside local and test environments"
                )
            if self.hsts_max_age_seconds <= 0:
                raise ValueError("HSTS is required outside local and test environments")
            if set(self.trusted_hosts).issubset(_LOCAL_TRUSTED_HOSTS):
                raise ValueError("an explicit deployment trusted host is required")
            if any(not origin.startswith("https://") for origin in self.cors_allowed_origins):
                raise ValueError("production CORS origins must use HTTPS")

        if not self.auth_enabled:
            return self

        if self.api_key_hmac_secret is None:
            raise ValueError("API-key HMAC secret is required when authentication is enabled")
        if len(self.api_key_hmac_secret.get_secret_value()) < 32:
            raise ValueError("API-key HMAC secret must contain at least 32 characters")
        if not self.api_key_digests:
            raise ValueError(
                "at least one API-key digest is required when authentication is enabled"
            )
        return self

    @model_validator(mode="after")
    def validate_processing_limits(self) -> Self:
        """Keep direct processing inside the accepted audio-duration boundary."""

        if self.sync_max_audio_duration_seconds > self.max_audio_duration_seconds:
            raise ValueError(
                "synchronous audio-duration limit cannot exceed the total duration limit"
            )
        if self.long_audio_queue_threshold_seconds > self.max_audio_duration_seconds:
            raise ValueError("long-audio queue threshold cannot exceed the total duration limit")
        if self.job_soft_time_limit_seconds >= self.job_hard_time_limit_seconds:
            raise ValueError("job soft time limit must be below the hard time limit")
        if self.artifact_cleanup_interval_seconds >= self.privacy_ttl_seconds:
            raise ValueError("artifact cleanup interval must be below the privacy TTL")
        if (
            self.async_jobs_enabled
            and self.job_hard_time_limit_seconds > self.private_artifact_ttl_seconds
        ):
            raise ValueError("job hard time limit cannot exceed private artifact retention")
        if (
            self.async_jobs_enabled
            and self.celery_visibility_timeout_seconds < self.job_hard_time_limit_seconds
        ):
            raise ValueError("Celery visibility timeout cannot be below the job hard time limit")
        if self.diarization_enabled and not self.async_jobs_enabled:
            raise ValueError("speaker diarization requires asynchronous jobs")
        if self.rate_limit_upload_requests > self.rate_limit_http_requests:
            raise ValueError("upload rate limit cannot exceed the general HTTP rate limit")
        if self.http_backlog < self.http_max_concurrency:
            raise ValueError("HTTP backlog cannot be below the HTTP concurrency limit")
        if self.live_max_session_seconds > self.privacy_ttl_seconds:
            raise ValueError("live session duration cannot exceed the privacy TTL")
        if self.live_max_utterance_seconds > self.live_max_session_seconds:
            raise ValueError("live utterance duration cannot exceed the session duration")
        if self.live_vad_analysis_window_seconds > self.live_max_utterance_seconds:
            raise ValueError("live VAD analysis window cannot exceed the utterance duration")
        if self.live_partial_min_audio_seconds > self.live_max_utterance_seconds:
            raise ValueError("live partial minimum cannot exceed the utterance duration")
        if self.live_vad_min_silence_ms > self.live_end_silence_ms:
            raise ValueError("VAD minimum silence cannot exceed utterance-end silence")
        if self.live_max_chunk_bytes % 2:
            raise ValueError("live PCM chunk limit must contain whole signed 16-bit samples")
        return self

    @property
    def private_artifact_ttl_seconds(self) -> int:
        """Expire private state early enough for the next cleanup run to meet the TTL."""

        return self.privacy_ttl_seconds - self.artifact_cleanup_interval_seconds


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
