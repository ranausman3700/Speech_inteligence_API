"""Celery implementation of the asynchronous job-dispatch boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import anyio
from celery import Celery

from speech_intelligence_api.adapters.observability import NoopObservability
from speech_intelligence_api.ports.observability import Observability

ENQUEUED_AT_HEADER = "speech-enqueued-at"


class CeleryJobDispatcher:
    """Publish opaque job IDs and keep private payloads out of the broker."""

    def __init__(
        self,
        app: Celery,
        *,
        task_name: str = "speech_intelligence.process_transcription",
        task_names_by_queue: dict[str, str] | None = None,
        observability: Observability | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._app = app
        self._task_name = task_name
        self._task_names_by_queue = dict(task_names_by_queue or {})
        self._observability = observability or NoopObservability()
        self._clock = clock

    async def enqueue(self, job_id: str, *, queue: str, priority: int) -> str:
        if not 0 <= priority <= 9:
            raise ValueError("job priority must be between 0 and 9")
        with self._observability.span(
            "speech.job.enqueue",
            kind="producer",
            attributes={"messaging.destination.name": queue},
        ) as span:
            headers = {ENQUEUED_AT_HEADER: f"{self._clock():.6f}"}
            self._observability.inject_trace_context(headers)
            try:
                result = await anyio.to_thread.run_sync(
                    partial(
                        self._app.send_task,
                        self._task_names_by_queue.get(queue, self._task_name),
                        args=[job_id],
                        queue=queue,
                        priority=priority,
                        headers=headers,
                    )
                )
            except Exception:
                span.mark_error()
                span.set_attribute("speech.job.outcome", "rejected")
                self._observability.record_job_submission(
                    queue=queue,
                    outcome="rejected",
                )
                raise
            span.set_attribute("speech.job.outcome", "submitted")
            self._observability.record_job_submission(
                queue=queue,
                outcome="submitted",
            )
            return str(result.id)

    async def revoke(self, task_id: str) -> None:
        await anyio.to_thread.run_sync(partial(self._app.control.revoke, task_id, terminate=False))
