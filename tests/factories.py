"""Reusable deterministic test object factories."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from speech_intelligence_api.config import Environment, Settings
from speech_intelligence_api.domain.enums import JobKind, JobStatus
from speech_intelligence_api.domain.models import JobRecord
from speech_intelligence_api.entrypoints.http.security import api_key_digest

TEST_API_KEY = "test-api-key-with-sufficient-entropy"
TEST_HMAC_SECRET = "test-hmac-secret-that-is-longer-than-32-characters"


def make_settings(
    *,
    auth_enabled: bool = True,
    environment: Environment = Environment.TEST,
    docs_enabled: bool = True,
) -> Settings:
    """Build settings without consulting the process environment."""

    values: dict[str, object] = {
        "environment": environment,
        "docs_enabled": docs_enabled,
        "auth_enabled": auth_enabled,
    }
    if auth_enabled:
        values.update(
            {
                "api_key_hmac_secret": TEST_HMAC_SECRET,
                "api_key_digests": (api_key_digest(TEST_API_KEY, TEST_HMAC_SECRET),),
            }
        )
    return Settings.model_validate(values)


def make_job(*, status: JobStatus = JobStatus.QUEUED) -> JobRecord:
    """Build an unexpired job with deterministic times."""

    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return JobRecord(
        job_id="job_01",
        kind=JobKind.TRANSCRIPTION,
        status=status,
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=30),
    )
