"""Privacy-safe metrics and distributed-tracing boundary."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias

SpanAttribute: TypeAlias = str | bool | int | float


class TelemetrySpan(Protocol):
    """Minimal mutable span surface used by transport and worker adapters."""

    def set_attribute(self, name: str, value: SpanAttribute) -> None:
        """Attach one bounded, non-sensitive attribute."""

    def update_name(self, name: str) -> None:
        """Replace the provisional span name with a route-based name."""

    def mark_error(self) -> None:
        """Mark the span as failed without recording exception text."""


class Observability(Protocol):
    """Record bounded operational telemetry without accepting customer content."""

    def span(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, SpanAttribute] | None = None,
        incoming_carrier: Mapping[str, str] | None = None,
    ) -> AbstractContextManager[TelemetrySpan]:
        """Create a privacy-safe span, optionally continuing a remote trace."""

    def inject_trace_context(self, carrier: MutableMapping[str, str]) -> None:
        """Inject only W3C trace-context headers into a private carrier."""

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Observe one HTTP request using its route template, never its raw path."""

    def record_job_submission(self, *, queue: str, outcome: str) -> None:
        """Count one background-job publication attempt."""

    def record_job_queue_wait(self, *, queue: str, duration_seconds: float) -> None:
        """Observe elapsed time between broker publication and worker start."""

    def record_job_processing(
        self,
        *,
        kind: str,
        queue: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        """Observe one worker-task execution."""

    def inference_started(self, *, operation: str, device: str) -> None:
        """Increment bounded in-flight inference state."""

    def inference_finished(
        self,
        *,
        operation: str,
        device: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        """Record inference duration and decrement in-flight state."""

    def render_metrics(self) -> tuple[bytes, str]:
        """Render the current Prometheus exposition payload and content type."""

    async def close(self) -> None:
        """Flush and close exporters owned by this process."""
