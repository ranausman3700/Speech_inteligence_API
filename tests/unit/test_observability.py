"""Privacy and contract tests for metrics and tracing adapters."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from speech_intelligence_api.adapters.observability import (
    NoopObservability,
    ServiceObservability,
    as_observability,
    build_observability,
)
from tests.factories import make_settings


@pytest.mark.asyncio
async def test_noop_observability_implements_the_complete_boundary() -> None:
    observability = NoopObservability()
    with observability.span(
        "ignored",
        kind="server",
        attributes={"safe": True},
        incoming_carrier={"traceparent": "ignored"},
    ) as span:
        span.set_attribute("ignored", "value")
        span.update_name("ignored-again")
        span.mark_error()

    carrier: dict[str, str] = {}
    observability.inject_trace_context(carrier)
    observability.record_http_request(
        method="GET",
        route="/health/live",
        status_code=200,
        duration_seconds=0.1,
    )
    observability.record_job_submission(queue="transcription.short", outcome="submitted")
    observability.record_job_queue_wait(queue="transcription.short", duration_seconds=0.2)
    observability.record_job_processing(
        kind="transcription",
        queue="transcription.short",
        outcome="completed",
        duration_seconds=0.3,
    )
    observability.inference_started(operation="transcription", device="cpu")
    observability.inference_finished(
        operation="transcription",
        device="cpu",
        outcome="completed",
        duration_seconds=0.4,
    )

    assert carrier == {}
    assert observability.render_metrics()[0] == b""
    assert as_observability(observability) is observability
    await observability.close()


@pytest.mark.asyncio
async def test_prometheus_metrics_use_only_bounded_operational_labels() -> None:
    private_value = "private-user@example.com"
    settings = make_settings().model_copy(update={"metrics_enabled": True})
    observability = ServiceObservability(settings)

    observability.record_http_request(
        method="GET",
        route="/v1/jobs/{job_id}",
        status_code=200,
        duration_seconds=0.25,
    )
    observability.record_http_request(
        method=private_value,
        route=private_value,
        status_code=999,
        duration_seconds=-1,
    )
    observability.record_job_submission(queue="transcription.short", outcome="submitted")
    observability.record_job_submission(queue=private_value, outcome=private_value)
    observability.record_job_queue_wait(queue="transcription.long", duration_seconds=2.5)
    observability.record_job_processing(
        kind="conversation",
        queue="diarization",
        outcome="completed",
        duration_seconds=10,
    )
    observability.record_job_processing(
        kind=private_value,
        queue=private_value,
        outcome=private_value,
        duration_seconds=-1,
    )
    observability.inference_started(operation="transcription", device="cpu")
    observability.inference_finished(
        operation="transcription",
        device="cpu",
        outcome="completed",
        duration_seconds=1.5,
    )
    observability.inference_started(operation=private_value, device=private_value)
    observability.inference_finished(
        operation=private_value,
        device=private_value,
        outcome=private_value,
        duration_seconds=-1,
    )

    payload, content_type = observability.render_metrics()
    text = payload.decode("utf-8")

    assert "speech_intelligence_http_requests_total" in text
    assert 'route="/v1/jobs/{job_id}"' in text
    assert 'method="OTHER",route="unmatched",status_code="500"' in text
    assert "speech_intelligence_job_queue_wait_seconds" in text
    assert "speech_intelligence_job_processing_duration_seconds" in text
    assert "speech_intelligence_inference_duration_seconds" in text
    assert 'device="cpu",operation="transcription"' in text
    assert 'device="unknown",operation="unknown"' in text
    assert private_value not in text
    assert content_type.startswith("text/plain")
    assert as_observability(observability) is observability
    await observability.close()


@pytest.mark.asyncio
async def test_w3c_trace_context_continues_without_exception_text() -> None:
    exporter = InMemorySpanExporter()
    settings = make_settings().model_copy(
        update={
            "tracing_enabled": True,
            "trace_sample_ratio": 1.0,
        }
    )
    observability = ServiceObservability(settings, span_exporter=exporter)
    carrier: dict[str, str] = {}

    with observability.span(
        "root",
        kind="server",
        attributes={"http.route": "/health/live"},
    ) as root:
        root.update_name("GET /health/live")
        observability.inject_trace_context(carrier)

    with observability.span(
        "speech.job.process",
        kind="consumer",
        incoming_carrier=carrier,
    ) as child:
        child.set_attribute("speech.job.kind", "transcription")
        child.mark_error()

    with (
        pytest.raises(RuntimeError, match="private failure text"),
        observability.span("sanitized-error", kind="unsupported"),
    ):
        raise RuntimeError("private failure text")

    await observability.close()
    spans = {span.name: span for span in exporter.get_finished_spans()}

    assert set(carrier) == {"traceparent"}
    assert spans["speech.job.process"].parent is not None
    assert spans["speech.job.process"].parent.span_id == spans["GET /health/live"].context.span_id
    assert spans["speech.job.process"].status.status_code is StatusCode.ERROR
    assert spans["sanitized-error"].status.status_code is StatusCode.ERROR
    assert spans["sanitized-error"].events == ()
    assert "private failure text" not in repr(spans)


@pytest.mark.asyncio
async def test_factory_builds_an_isolated_disabled_registry() -> None:
    observability = build_observability(make_settings())

    payload, _ = observability.render_metrics()

    assert payload == b""
    await observability.close()
