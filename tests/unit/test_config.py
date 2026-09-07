"""Settings validation tests."""

import pytest
from pydantic import ValidationError

from speech_intelligence_api.config import Environment, Settings, get_settings
from tests.factories import TEST_HMAC_SECRET


def test_local_defaults_are_runnable_without_credentials() -> None:
    settings = Settings.model_validate({})

    assert settings.environment is Environment.LOCAL
    assert settings.auth_enabled is False
    assert settings.metrics_enabled is False
    assert settings.tracing_enabled is False
    assert settings.trace_sample_ratio == 0.10
    assert settings.privacy_ttl_seconds == 1800
    assert settings.private_artifact_ttl_seconds == 1770
    assert settings.http_max_concurrency >= 1000
    assert settings.http_backlog >= settings.http_max_concurrency
    assert settings.trusted_hosts == ("localhost", "127.0.0.1", "testserver")
    assert settings.security_headers_enabled is True
    assert settings.hsts_max_age_seconds == 31_536_000


def test_production_rejects_disabled_authentication() -> None:
    with pytest.raises(ValidationError, match="authentication is required"):
        Settings.model_validate({"environment": Environment.PRODUCTION})


def test_production_rejects_disabled_rate_limiting() -> None:
    with pytest.raises(ValidationError, match="rate limiting is required"):
        Settings.model_validate(
            {
                "environment": Environment.PRODUCTION,
                "auth_enabled": True,
                "api_key_hmac_secret": TEST_HMAC_SECRET,
                "api_key_digests": ("a" * 64,),
            }
        )


def _secure_production_values() -> dict[str, object]:
    return {
        "environment": Environment.PRODUCTION,
        "auth_enabled": True,
        "api_key_hmac_secret": TEST_HMAC_SECRET,
        "api_key_digests": ("a" * 64,),
        "rate_limit_enabled": True,
        "trusted_hosts": ("api.example.com", "127.0.0.1"),
    }


def test_production_requires_explicit_deployment_host() -> None:
    values = _secure_production_values()
    values["trusted_hosts"] = ("localhost", "127.0.0.1")

    with pytest.raises(ValidationError, match="deployment trusted host"):
        Settings.model_validate(values)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"security_headers_enabled": False}, "security headers"),
        ({"hsts_max_age_seconds": 0}, "HSTS"),
        ({"cors_allowed_origins": ("http://app.example.com",)}, "must use HTTPS"),
    ],
)
def test_production_http_boundary_fails_closed(
    override: dict[str, object],
    message: str,
) -> None:
    values = _secure_production_values()
    values.update(override)

    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(values)


@pytest.mark.parametrize(
    ("secret", "digests", "message"),
    [
        (None, ("0" * 64,), "HMAC secret is required"),
        ("too-short", ("0" * 64,), "at least 32 characters"),
        (TEST_HMAC_SECRET, (), "at least one API-key digest"),
        (TEST_HMAC_SECRET, ("not-a-digest",), "64 lowercase hexadecimal"),
        (TEST_HMAC_SECRET, ("a" * 64, "a" * 64), "must be unique"),
    ],
)
def test_enabled_authentication_requires_valid_key_material(
    secret: str | None,
    digests: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(
            {
                "environment": Environment.TEST,
                "auth_enabled": True,
                "api_key_hmac_secret": secret,
                "api_key_digests": digests,
            }
        )


def test_secret_is_redacted_from_settings_representation() -> None:
    settings = Settings.model_validate(
        {
            "auth_enabled": True,
            "api_key_hmac_secret": TEST_HMAC_SECRET,
            "api_key_digests": ("a" * 64,),
        }
    )

    assert TEST_HMAC_SECRET not in repr(settings)
    assert "**********" in repr(settings.api_key_hmac_secret)


def test_invalid_request_id_header_is_rejected() -> None:
    with pytest.raises(ValidationError, match="valid HTTP field name"):
        Settings.model_validate({"request_id_header": "bad header"})


@pytest.mark.parametrize(
    "hosts",
    [
        (),
        ("*",),
        ("*.example.com",),
        ("https://api.example.com",),
        ("api.example.com:8000",),
        ("api.example.com", "API.EXAMPLE.COM."),
    ],
)
def test_trusted_hosts_must_be_explicit_and_unique(hosts: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="trusted host"):
        Settings.model_validate({"trusted_hosts": hosts})


def test_trusted_hosts_are_normalized() -> None:
    settings = Settings.model_validate({"trusted_hosts": ("API.Example.COM.",)})

    assert settings.trusted_hosts == ("api.example.com",)


@pytest.mark.parametrize(
    "origins",
    [
        ("*",),
        ("https://example.com/path",),
        ("https://example.com", "https://example.com/"),
    ],
)
def test_cors_origins_must_be_explicit_and_unique(origins: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="CORS origins"):
        Settings.model_validate({"cors_allowed_origins": origins})


def test_cors_origins_are_normalized() -> None:
    settings = Settings.model_validate({"cors_allowed_origins": ("https://app.example.com/",)})

    assert settings.cors_allowed_origins == ("https://app.example.com",)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "max_audio_duration_seconds": 60,
                "sync_max_audio_duration_seconds": 50,
                "long_audio_queue_threshold_seconds": 61,
            },
            "long-audio queue threshold",
        ),
        (
            {
                "job_soft_time_limit_seconds": 100,
                "job_hard_time_limit_seconds": 100,
            },
            "soft time limit",
        ),
        (
            {
                "async_jobs_enabled": True,
                "privacy_ttl_seconds": 300,
                "job_soft_time_limit_seconds": 200,
                "job_hard_time_limit_seconds": 300,
            },
            "cannot exceed private artifact retention",
        ),
        (
            {
                "async_jobs_enabled": True,
                "job_soft_time_limit_seconds": 100,
                "job_hard_time_limit_seconds": 200,
                "celery_visibility_timeout_seconds": 199,
            },
            "visibility timeout",
        ),
    ],
)
def test_async_processing_limits_fail_closed(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"privacy_ttl_seconds": 60, "live_max_session_seconds": 61},
            "session duration",
        ),
        (
            {"live_max_session_seconds": 20, "live_max_utterance_seconds": 21},
            "utterance duration",
        ),
        (
            {
                "live_max_utterance_seconds": 2,
                "live_vad_analysis_window_seconds": 3,
            },
            "analysis window",
        ),
        (
            {
                "live_max_utterance_seconds": 2,
                "live_vad_analysis_window_seconds": 1,
                "live_partial_min_audio_seconds": 3,
            },
            "partial minimum",
        ),
        (
            {"live_vad_min_silence_ms": 800, "live_end_silence_ms": 700},
            "minimum silence",
        ),
        (
            {"live_max_chunk_bytes": 1025},
            "whole signed 16-bit",
        ),
    ],
)
def test_live_processing_limits_fail_closed(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(values)


def test_upload_rate_limit_cannot_exceed_general_http_limit() -> None:
    with pytest.raises(ValidationError, match="upload rate limit"):
        Settings.model_validate(
            {
                "rate_limit_http_requests": 10,
                "rate_limit_upload_requests": 11,
            }
        )


def test_http_backlog_cannot_be_below_concurrency_limit() -> None:
    with pytest.raises(ValidationError, match="HTTP backlog"):
        Settings.model_validate(
            {
                "http_max_concurrency": 1500,
                "http_backlog": 1200,
            }
        )


def test_cleanup_interval_must_leave_positive_private_retention() -> None:
    with pytest.raises(ValidationError, match="cleanup interval"):
        Settings.model_validate(
            {
                "privacy_ttl_seconds": 60,
                "artifact_cleanup_interval_seconds": 60,
            }
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://collector.example.com/v1/traces",
        "http://user:password@collector.example.com/v1/traces",
        "https://collector.example.com/v1/traces?private=value",
        "https://collector.example.com/v1/traces#fragment",
    ],
)
def test_otlp_trace_endpoint_must_be_an_explicit_http_url(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="OTLP traces endpoint"):
        Settings.model_validate({"otlp_traces_endpoint": endpoint})


def test_https_otlp_trace_endpoint_is_accepted() -> None:
    settings = Settings.model_validate(
        {"otlp_traces_endpoint": "https://collector.example.com/v1/traces"}
    )

    assert settings.otlp_traces_endpoint == "https://collector.example.com/v1/traces"


def test_settings_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_API_ENVIRONMENT", "local")
    monkeypatch.setenv("SPEECH_API_AUTH_ENABLED", "false")
    get_settings.cache_clear()

    assert get_settings().environment is Environment.LOCAL

    get_settings.cache_clear()
