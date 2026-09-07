"""Celery task entrypoints with bounded retries and sanitized failure state."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

from celery import Task
from celery.signals import worker_process_shutdown
from redis.exceptions import RedisError

from speech_intelligence_api.adapters.celery_jobs import ENQUEUED_AT_HEADER
from speech_intelligence_api.adapters.observability import build_observability
from speech_intelligence_api.bootstrap import (
    ConversationWorkerRuntime,
    WorkerRuntime,
    build_conversation_worker_runtime,
    build_worker_runtime,
)
from speech_intelligence_api.config import get_settings
from speech_intelligence_api.domain.enums import JobQueue
from speech_intelligence_api.domain.errors import (
    DependencyUnavailableError,
    DiarizationUnavailableError,
    ErrorCode,
    ModelUnavailableError,
    ServiceError,
)
from speech_intelligence_api.ports.observability import Observability, TelemetrySpan
from speech_intelligence_api.workers.celery_app import (
    CLEANUP_TASK_NAME,
    CONVERSATION_TASK_NAME,
    TRANSCRIPTION_TASK_NAME,
    celery_app,
)

_ResultT = TypeVar("_ResultT")


@lru_cache(maxsize=1)
def _runtime() -> WorkerRuntime:
    settings = get_settings()
    if not settings.async_jobs_enabled:
        raise RuntimeError("asynchronous jobs are disabled")
    return build_worker_runtime(settings, observability=_observability())


@lru_cache(maxsize=1)
def _conversation_runtime() -> ConversationWorkerRuntime:
    settings = get_settings()
    if not settings.async_jobs_enabled or not settings.diarization_enabled:
        raise RuntimeError("speaker diarization jobs are disabled")
    return build_conversation_worker_runtime(settings, observability=_observability())


@lru_cache(maxsize=1)
def _observability() -> Observability:
    """Create one telemetry registry/exporter per worker process."""

    return build_observability(get_settings())


@lru_cache(maxsize=1)
def _event_loop() -> asyncio.AbstractEventLoop:
    """Keep async Redis connections on one loop for the worker-process lifetime."""

    return asyncio.new_event_loop()


def _run(awaitable: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    return _event_loop().run_until_complete(awaitable)


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name=TRANSCRIPTION_TASK_NAME,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_transcription(self: Task, job_id: str) -> None:
    """Execute one opaque job ID and retry transient dependency failures."""

    settings = get_settings()
    with _observe_task(
        self,
        kind="transcription",
        fallback_queue=JobQueue.SHORT_TRANSCRIPTION.value,
    ) as observation:
        try:
            _run(_runtime().service.execute(job_id))
            observation.outcome = "completed"
        except (ModelUnavailableError, RedisError, OSError):
            if self.request.retries < settings.job_max_retries:
                observation.outcome = "retrying"
                countdown = settings.job_retry_backoff_seconds * (2**self.request.retries)
                raise self.retry(
                    exc=DependencyUnavailableError(),
                    countdown=countdown,
                    max_retries=settings.job_max_retries,
                ) from None
            _run(
                _runtime().service.fail(
                    job_id,
                    ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                )
            )
        except ServiceError as exc:
            _run(_runtime().service.fail(job_id, exc.code.value))
        except Exception:
            _run(_runtime().service.fail(job_id, ErrorCode.INTERNAL_ERROR.value))
            raise


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name=CONVERSATION_TASK_NAME,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_conversation(self: Task, job_id: str) -> None:
    """Execute one opaque conversation job with bounded transient retries."""

    settings = get_settings()
    with _observe_task(
        self,
        kind="conversation",
        fallback_queue=JobQueue.DIARIZATION.value,
    ) as observation:
        try:
            _run(_conversation_runtime().service.execute(job_id))
            observation.outcome = "completed"
        except (DiarizationUnavailableError, ModelUnavailableError, RedisError, OSError):
            if self.request.retries < settings.job_max_retries:
                observation.outcome = "retrying"
                countdown = settings.job_retry_backoff_seconds * (2**self.request.retries)
                raise self.retry(
                    exc=DependencyUnavailableError(),
                    countdown=countdown,
                    max_retries=settings.job_max_retries,
                ) from None
            _run(
                _conversation_runtime().service.fail(
                    job_id,
                    ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                )
            )
        except ServiceError as exc:
            _run(_conversation_runtime().service.fail(job_id, exc.code.value))
        except Exception:
            _run(_conversation_runtime().service.fail(job_id, ErrorCode.INTERNAL_ERROR.value))
            raise


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name=CLEANUP_TASK_NAME,
    ignore_result=True,
)
def cleanup_expired_artifacts(self: Task) -> None:
    """Delete expired private data with bounded sanitized retries."""

    settings = get_settings()
    with _observe_task(self, kind="cleanup", fallback_queue=JobQueue.CLEANUP.value) as observation:
        try:
            _run(_runtime().cleanup_service.execute())
            observation.outcome = "completed"
        except DependencyUnavailableError:
            if self.request.retries < settings.job_max_retries:
                observation.outcome = "retrying"
                countdown = settings.job_retry_backoff_seconds * (2**self.request.retries)
                raise self.retry(
                    exc=DependencyUnavailableError(),
                    countdown=countdown,
                    max_retries=settings.job_max_retries,
                ) from None
            raise


@worker_process_shutdown.connect  # type: ignore[untyped-decorator]
def close_worker_process_resources(**_: object) -> None:
    """Close cached event-loop resources when a prefork child exits."""

    if (
        _runtime.cache_info().currsize == 0
        and _conversation_runtime.cache_info().currsize == 0
        and _observability.cache_info().currsize == 0
    ):
        return
    loop = _event_loop()
    if not loop.is_closed():
        if _runtime.cache_info().currsize:
            loop.run_until_complete(_runtime().redis_client.aclose())
        if _conversation_runtime.cache_info().currsize:
            loop.run_until_complete(_conversation_runtime().redis_client.aclose())
        if _observability.cache_info().currsize:
            loop.run_until_complete(_observability().close())
        loop.close()
    _runtime.cache_clear()
    _conversation_runtime.cache_clear()
    _observability.cache_clear()
    _event_loop.cache_clear()


@dataclass(slots=True)
class _TaskObservation:
    span: TelemetrySpan
    outcome: str = "failed"


@contextmanager
def _observe_task(
    task: Task,
    *,
    kind: str,
    fallback_queue: str,
) -> Iterator[_TaskObservation]:
    observability = _observability()
    headers = _request_headers(task)
    queue = _request_queue(task, fallback_queue)
    enqueued_at = _enqueued_at(headers)
    if enqueued_at is not None:
        observability.record_job_queue_wait(
            queue=queue,
            duration_seconds=max(0.0, time.time() - enqueued_at),
        )
    started_at = time.perf_counter()
    with observability.span(
        "speech.job.process",
        kind="consumer",
        attributes={
            "messaging.destination.name": queue,
            "speech.job.kind": kind,
        },
        incoming_carrier=headers,
    ) as span:
        observation = _TaskObservation(span=span)
        try:
            yield observation
        finally:
            span.set_attribute("speech.job.outcome", observation.outcome)
            if observation.outcome == "failed":
                span.mark_error()
            observability.record_job_processing(
                kind=kind,
                queue=queue,
                outcome=observation.outcome,
                duration_seconds=time.perf_counter() - started_at,
            )


def _request_headers(task: Task) -> dict[str, str]:
    raw_headers = getattr(task.request, "headers", None)
    if not isinstance(raw_headers, Mapping):
        return {}
    allowed = {ENQUEUED_AT_HEADER, "traceparent"}
    return {
        str(name).lower(): str(value)
        for name, value in raw_headers.items()
        if str(name).lower() in allowed
    }


def _request_queue(task: Task, fallback: str) -> str:
    delivery_info = getattr(task.request, "delivery_info", None)
    if isinstance(delivery_info, Mapping):
        routing_key = delivery_info.get("routing_key")
        if isinstance(routing_key, str) and routing_key in {queue.value for queue in JobQueue}:
            return routing_key
    return fallback


def _enqueued_at(headers: Mapping[str, str]) -> float | None:
    try:
        value = float(headers[ENQUEUED_AT_HEADER])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None
