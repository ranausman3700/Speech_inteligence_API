"""FastAPI integration tests."""

import asyncio
import re
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from speech_intelligence_api.adapters.observability import ServiceObservability
from speech_intelligence_api.application.readiness import ReadinessCheck
from speech_intelligence_api.application.transcriptions import BatchTranscriptionService
from speech_intelligence_api.domain.enums import SUPPORTED_LANGUAGE_VARIANTS, LanguageCode
from speech_intelligence_api.entrypoints.http.app import create_app
from speech_intelligence_api.ports.rate_limiting import RateLimitDecision
from tests.factories import TEST_API_KEY, make_settings


class _CountingRateLimiter:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.calls: list[tuple[str, str, int, int]] = []

    async def acquire(
        self,
        identity: str,
        *,
        scope: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        self.calls.append((identity, scope, limit, window_seconds))
        key = (identity, scope)
        count = self.counts.get(key, 0)
        allowed = count < limit
        if allowed:
            self.counts[key] = count + 1
        return RateLimitDecision(
            allowed=allowed,
            limit=limit,
            remaining=max(0, limit - count - 1) if allowed else 0,
            retry_after_seconds=0 if allowed else 7,
            reset_after_seconds=7,
        )

    async def ping(self) -> None:
        return None


def _rate_limited_app(
    limiter: _CountingRateLimiter,
    *,
    auth_enabled: bool = True,
) -> FastAPI:
    settings = make_settings(auth_enabled=auth_enabled).model_copy(
        update={
            "rate_limit_enabled": True,
            "rate_limit_http_requests": 1,
            "rate_limit_upload_requests": 1,
            "rate_limit_live_sessions": 1,
            "rate_limit_window_seconds": 60,
        }
    )
    return create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, object()),
        rate_limiter=limiter,
    )


def test_health_endpoints_are_public(client: TestClient) -> None:
    live_response = client.get("/health/live")
    ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json()["status"] == "ok"
    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "service": "speech-intelligence-api",
        "version": "0.1.0",
        "checks": {"application": "ok"},
    }


def test_every_http_response_has_private_api_security_headers(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "camera=(), geolocation=(), microphone=()"
    assert "strict-transport-security" not in response.headers


def test_https_response_advertises_hsts() -> None:
    app = create_app(make_settings())

    with TestClient(app, base_url="https://testserver") as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == "max-age=31536000"


def test_untrusted_host_is_rejected_with_security_headers() -> None:
    app = create_app(make_settings())

    with TestClient(app) as client:
        response = client.get("/health/live", headers={"Host": "attacker.example"})

    assert response.status_code == 400
    assert response.text == "Invalid host header"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_security_headers_can_be_disabled_only_for_nonproduction() -> None:
    settings = make_settings().model_copy(update={"security_headers_enabled": False})
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert "x-content-type-options" not in response.headers
    assert "cache-control" not in response.headers


def test_metrics_endpoint_is_opt_in_authenticated_and_privacy_safe() -> None:
    settings = make_settings().model_copy(update={"metrics_enabled": True})
    app = create_app(settings)

    with TestClient(app) as client:
        unauthorized = client.get("/metrics")
        client.get("/health/live")
        response = client.get("/metrics", headers={"X-API-Key": TEST_API_KEY})

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("text/plain")
    assert "speech_intelligence_http_requests_total" in response.text
    assert 'route="/health/live"' in response.text
    assert TEST_API_KEY not in response.text


def test_metrics_endpoint_is_absent_when_disabled(client: TestClient) -> None:
    response = client.get("/metrics", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 404


def test_http_middleware_continues_w3c_trace_using_only_route_metadata() -> None:
    exporter = InMemorySpanExporter()
    settings = make_settings().model_copy(
        update={"tracing_enabled": True, "trace_sample_ratio": 1.0}
    )
    observability = ServiceObservability(settings, span_exporter=exporter)
    app = create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, object()),
        observability=observability,
    )
    remote_trace_id = "1" * 32
    remote_parent_id = "2" * 16

    with TestClient(app) as client:
        response = client.get(
            "/health/live?private-query=do-not-export",
            headers={
                "traceparent": f"00-{remote_trace_id}-{remote_parent_id}-01",
                "X-API-Key": "private-key-do-not-export",
            },
        )

    asyncio.run(observability.close())
    spans = exporter.get_finished_spans()
    server_span = next(span for span in spans if span.name == "GET /health/live")

    assert response.status_code == 200
    assert f"{server_span.context.trace_id:032x}" == remote_trace_id
    assert server_span.parent is not None
    assert f"{server_span.parent.span_id:016x}" == remote_parent_id
    attributes = server_span.attributes
    assert attributes is not None
    assert attributes["http.route"] == "/health/live"
    assert "private-query" not in repr(attributes)
    assert "private-key" not in repr(attributes)


def test_capabilities_require_api_key(client: TestClient) -> None:
    response = client.get("/v1/capabilities")

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["www-authenticate"] == "ApiKey"
    assert response.json()["code"] == "authentication_failed"


def test_capabilities_return_native_script_contract(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/v1/capabilities", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcription_task"] == "transcribe"
    assert payload["translation_enabled"] is False
    assert len(payload["languages"]) == len(SUPPORTED_LANGUAGE_VARIANTS)
    assert {item["code"] for item in payload["languages"]} == {code.value for code in LanguageCode}
    assert {item["chinese_script"] for item in payload["languages"] if item["code"] == "zh"} == {
        "simplified",
        "traditional",
    }


def test_invalid_api_key_never_appears_in_response(client: TestClient) -> None:
    response = client.get("/v1/capabilities", headers={"X-API-Key": "do-not-leak-me"})

    assert response.status_code == 401
    assert "do-not-leak-me" not in response.text


def test_authenticated_http_rate_limit_returns_headers_and_problem_contract() -> None:
    limiter = _CountingRateLimiter()
    app = _rate_limited_app(limiter)

    with TestClient(app) as client:
        allowed = client.get("/v1/capabilities", headers={"X-API-Key": TEST_API_KEY})
        denied = client.get("/v1/capabilities", headers={"X-API-Key": TEST_API_KEY})

    assert allowed.status_code == 200
    assert allowed.headers["x-ratelimit-limit"] == "1"
    assert allowed.headers["x-ratelimit-remaining"] == "0"
    assert denied.status_code == 429
    assert denied.headers["retry-after"] == "7"
    assert denied.headers["x-ratelimit-remaining"] == "0"
    assert denied.json()["code"] == "rate_limited"
    assert denied.json()["details"] == {
        "limit": 1,
        "window_seconds": 60,
        "retry_after_seconds": 7,
    }
    assert TEST_API_KEY not in denied.text


def test_rate_limit_uses_ip_fallback_and_separate_upload_policy() -> None:
    limiter = _CountingRateLimiter()
    app = _rate_limited_app(limiter, auth_enabled=False)

    with TestClient(app) as client:
        client.get("/v1/capabilities")
        response = client.post("/v1/transcriptions")

    assert response.status_code == 422
    assert limiter.calls[0] == ("ip:testclient", "http", 1, 60)
    assert limiter.calls[1] == ("ip:testclient", "upload", 1, 60)


def test_enabled_rate_limit_fails_closed_when_not_composed() -> None:
    settings = make_settings().model_copy(update={"rate_limit_enabled": True})
    app = create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, object()),
    )

    with TestClient(app) as client:
        response = client.get("/v1/capabilities", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 503
    assert response.json()["code"] == "dependency_unavailable"


def test_safe_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "caller-id:123"})

    assert response.headers["x-request-id"] == "caller-id:123"
    assert response.json()["status"] == "ok"


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "bad request id"})

    request_id = response.headers["x-request-id"]
    assert request_id != "bad request id"
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)


class _FailingCheck:
    @property
    def name(self) -> str:
        return "redis"

    async def check(self) -> None:
        raise ConnectionError("internal connection information")


def test_readiness_reports_dependency_failure_without_details() -> None:
    check: ReadinessCheck = _FailingCheck()
    app = create_app(make_settings(), readiness_checks=[check])

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["redis"] == "unavailable"
    assert "internal connection information" not in response.text


def test_unexpected_exception_is_sanitized() -> None:
    app = create_app(make_settings(auth_enabled=False))

    @app.get("/explode")
    async def explode() -> None:
        raise RuntimeError("sensitive internal value")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/explode")

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "sensitive internal value" not in response.text


def test_validation_error_does_not_echo_invalid_input() -> None:
    app = create_app(make_settings(auth_enabled=False))

    @app.get("/items/{item_id}")
    async def item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    with TestClient(app) as client:
        response = client.get("/items/not-a-private-id")

    assert response.status_code == 422
    assert response.json()["issues"] == [{"location": "path.item_id", "error_type": "int_parsing"}]
    assert "not-a-private-id" not in response.text


def test_unknown_route_uses_problem_contract(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert response.json()["trace_id"] == response.headers["x-request-id"]


def test_version_mismatch_fails_application_creation() -> None:
    settings = make_settings().model_copy(update={"service_version": "9.9.9"})

    try:
        create_app(settings)
    except ValueError as exc:
        assert str(exc) == "configured service version must match the package version"
    else:
        raise AssertionError("version mismatch did not fail")


def test_app_fixture_is_fastapi(app: FastAPI) -> None:
    assert isinstance(app, FastAPI)
    assert TEST_API_KEY not in repr(app.state.authenticator)
