"""Prometheus metrics and OpenTelemetry tracing with bounded metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from functools import partial
from typing import Final

import anyio
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.enums import JobQueue
from speech_intelligence_api.ports.observability import (
    Observability,
    SpanAttribute,
    TelemetrySpan,
)

_HTTP_METHODS: Final = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})
_QUEUES: Final = frozenset(queue.value for queue in JobQueue)
_JOB_KINDS: Final = frozenset({"cleanup", "conversation", "transcription"})
_OUTCOMES: Final = frozenset({"completed", "failed", "rejected", "retrying", "submitted"})
_INFERENCE_OPERATIONS: Final = frozenset({"diarization", "transcription"})
_DEVICES: Final = frozenset({"auto", "cpu", "cuda"})
_SPAN_KINDS: Final = {
    "client": SpanKind.CLIENT,
    "consumer": SpanKind.CONSUMER,
    "internal": SpanKind.INTERNAL,
    "producer": SpanKind.PRODUCER,
    "server": SpanKind.SERVER,
}


class _NoopSpan:
    def set_attribute(self, name: str, value: SpanAttribute) -> None:
        del name, value

    def update_name(self, name: str) -> None:
        del name

    def mark_error(self) -> None:
        return None


class NoopObservability:
    """Zero-cost behavior used when observability is not composed."""

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, SpanAttribute] | None = None,
        incoming_carrier: Mapping[str, str] | None = None,
    ) -> Iterator[TelemetrySpan]:
        del name, kind, attributes, incoming_carrier
        yield _NoopSpan()

    def inject_trace_context(self, carrier: MutableMapping[str, str]) -> None:
        del carrier

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        del method, route, status_code, duration_seconds

    def record_job_submission(self, *, queue: str, outcome: str) -> None:
        del queue, outcome

    def record_job_queue_wait(self, *, queue: str, duration_seconds: float) -> None:
        del queue, duration_seconds

    def record_job_processing(
        self,
        *,
        kind: str,
        queue: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        del kind, queue, outcome, duration_seconds

    def inference_started(self, *, operation: str, device: str) -> None:
        del operation, device

    def inference_finished(
        self,
        *,
        operation: str,
        device: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        del operation, device, outcome, duration_seconds

    def render_metrics(self) -> tuple[bytes, str]:
        return b"", CONTENT_TYPE_LATEST

    async def close(self) -> None:
        return None


class _OtelSpan:
    def __init__(self, span: Span) -> None:
        self._span = span

    def set_attribute(self, name: str, value: SpanAttribute) -> None:
        self._span.set_attribute(name, value)

    def update_name(self, name: str) -> None:
        self._span.update_name(name)

    def mark_error(self) -> None:
        self._span.set_status(Status(StatusCode.ERROR))


class ServiceObservability:
    """Per-process telemetry registry with an optional OTLP trace exporter."""

    def __init__(
        self,
        settings: Settings,
        *,
        span_exporter: SpanExporter | None = None,
    ) -> None:
        self._metrics_enabled = settings.metrics_enabled
        self._allowed_routes = _allowed_routes(settings.api_prefix)
        self._registry = CollectorRegistry(auto_describe=True)
        self._provider: TracerProvider | None = None
        self._propagator = TraceContextTextMapPropagator()

        if settings.tracing_enabled:
            exporter = span_exporter or self._otlp_exporter(settings)
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": settings.service_name,
                        "service.version": settings.service_version,
                        "deployment.environment.name": settings.environment.value,
                    }
                ),
                sampler=ParentBased(TraceIdRatioBased(settings.trace_sample_ratio)),
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            self._provider = provider
            self._tracer = provider.get_tracer(
                "speech_intelligence_api",
                settings.service_version,
            )
        else:
            self._tracer = trace.NoOpTracerProvider().get_tracer("speech_intelligence_api")

        self._http_requests: Counter | None = None
        self._http_duration: Histogram | None = None
        self._job_submissions: Counter | None = None
        self._job_queue_wait: Histogram | None = None
        self._job_processing: Histogram | None = None
        self._inference_duration: Histogram | None = None
        self._inference_in_progress: Gauge | None = None
        if self._metrics_enabled:
            self._configure_metrics(settings)

    @staticmethod
    def _otlp_exporter(settings: Settings) -> SpanExporter:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(
            endpoint=settings.otlp_traces_endpoint,
            timeout=settings.trace_export_timeout_seconds,
        )

    def _configure_metrics(self, settings: Settings) -> None:
        Gauge(
            "speech_intelligence_build_info",
            "Static build and deployment information.",
            ("environment", "version"),
            registry=self._registry,
        ).labels(
            environment=settings.environment.value,
            version=settings.service_version,
        ).set(1)
        self._http_requests = Counter(
            "speech_intelligence_http_requests_total",
            "Completed HTTP requests by route template and status.",
            ("method", "route", "status_code"),
            registry=self._registry,
        )
        self._http_duration = Histogram(
            "speech_intelligence_http_request_duration_seconds",
            "HTTP request duration by route template.",
            ("method", "route"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self._registry,
        )
        self._job_submissions = Counter(
            "speech_intelligence_job_submissions_total",
            "Background job publication attempts.",
            ("queue", "outcome"),
            registry=self._registry,
        )
        self._job_queue_wait = Histogram(
            "speech_intelligence_job_queue_wait_seconds",
            "Elapsed time from broker publication to worker start.",
            ("queue",),
            buckets=(0.01, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800),
            registry=self._registry,
        )
        self._job_processing = Histogram(
            "speech_intelligence_job_processing_duration_seconds",
            "Worker task duration by bounded job metadata.",
            ("kind", "queue", "outcome"),
            buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200, 1800),
            registry=self._registry,
        )
        self._inference_duration = Histogram(
            "speech_intelligence_inference_duration_seconds",
            "Model inference duration without customer-content labels.",
            ("operation", "device", "outcome"),
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
            registry=self._registry,
        )
        self._inference_in_progress = Gauge(
            "speech_intelligence_inference_in_progress",
            "Model inference calls currently in progress.",
            ("operation", "device"),
            registry=self._registry,
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, SpanAttribute] | None = None,
        incoming_carrier: Mapping[str, str] | None = None,
    ) -> Iterator[TelemetrySpan]:
        parent_context = (
            self._propagator.extract(carrier=dict(incoming_carrier))
            if incoming_carrier is not None
            else None
        )
        with self._tracer.start_as_current_span(
            name,
            context=parent_context,
            kind=_SPAN_KINDS.get(kind, SpanKind.INTERNAL),
            record_exception=False,
            set_status_on_exception=False,
        ) as raw_span:
            span = _OtelSpan(raw_span)
            for attribute_name, value in (attributes or {}).items():
                span.set_attribute(attribute_name, value)
            try:
                yield span
            except Exception:
                span.mark_error()
                raise

    def inject_trace_context(self, carrier: MutableMapping[str, str]) -> None:
        self._propagator.inject(carrier=carrier)

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        if self._http_requests is None or self._http_duration is None:
            return
        labels = {"method": _method(method), "route": _route(route, self._allowed_routes)}
        self._http_requests.labels(**labels, status_code=_status_code(status_code)).inc()
        self._http_duration.labels(**labels).observe(_duration(duration_seconds))

    def record_job_submission(self, *, queue: str, outcome: str) -> None:
        if self._job_submissions is not None:
            self._job_submissions.labels(
                queue=_bounded(queue, _QUEUES),
                outcome=_bounded(outcome, _OUTCOMES),
            ).inc()

    def record_job_queue_wait(self, *, queue: str, duration_seconds: float) -> None:
        if self._job_queue_wait is not None:
            self._job_queue_wait.labels(queue=_bounded(queue, _QUEUES)).observe(
                _duration(duration_seconds)
            )

    def record_job_processing(
        self,
        *,
        kind: str,
        queue: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        if self._job_processing is not None:
            self._job_processing.labels(
                kind=_bounded(kind, _JOB_KINDS),
                queue=_bounded(queue, _QUEUES),
                outcome=_bounded(outcome, _OUTCOMES),
            ).observe(_duration(duration_seconds))

    def inference_started(self, *, operation: str, device: str) -> None:
        if self._inference_in_progress is not None:
            self._inference_in_progress.labels(
                operation=_bounded(operation, _INFERENCE_OPERATIONS),
                device=_bounded(device, _DEVICES),
            ).inc()

    def inference_finished(
        self,
        *,
        operation: str,
        device: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        labels = {
            "operation": _bounded(operation, _INFERENCE_OPERATIONS),
            "device": _bounded(device, _DEVICES),
        }
        if self._inference_in_progress is not None:
            self._inference_in_progress.labels(**labels).dec()
        if self._inference_duration is not None:
            self._inference_duration.labels(
                **labels,
                outcome=_bounded(outcome, _OUTCOMES),
            ).observe(_duration(duration_seconds))

    def render_metrics(self) -> tuple[bytes, str]:
        return generate_latest(self._registry), CONTENT_TYPE_LATEST

    async def close(self) -> None:
        if self._provider is not None:
            await anyio.to_thread.run_sync(partial(self._provider.shutdown))


def build_observability(settings: Settings) -> ServiceObservability:
    """Compose one isolated metrics registry and trace provider per process."""

    return ServiceObservability(settings)


def _method(value: str) -> str:
    normalized = value.upper()
    return normalized if normalized in _HTTP_METHODS else "OTHER"


def _route(value: str, allowed: frozenset[str]) -> str:
    return value if value in allowed else "unmatched"


def _allowed_routes(api_prefix: str) -> frozenset[str]:
    return frozenset(
        {
            "/docs",
            "/docs/oauth2-redirect",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/openapi.json",
            f"{api_prefix}/capabilities",
            f"{api_prefix}/conversations",
            f"{api_prefix}/jobs/{{job_id}}",
            f"{api_prefix}/jobs/{{job_id}}/cancel",
            f"{api_prefix}/jobs/{{job_id}}/result",
            f"{api_prefix}/live-transcription",
            f"{api_prefix}/transcriptions",
        }
    )


def _status_code(value: int) -> str:
    return str(value) if 100 <= value <= 599 else "500"


def _bounded(value: str, allowed: frozenset[str]) -> str:
    return value if value in allowed else "unknown"


def _duration(value: float) -> float:
    return max(0.0, value)


def as_observability(value: ServiceObservability | NoopObservability) -> Observability:
    """Provide an explicit structural-typing assertion for composition sites."""

    return value
