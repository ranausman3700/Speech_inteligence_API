"""Celery task-wrapper retry, failure, and cleanup tests."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import cast

import pytest
from celery import Task
from celery.exceptions import Retry

from speech_intelligence_api.adapters.observability import ServiceObservability
from speech_intelligence_api.application.privacy import PrivacyCleanupService
from speech_intelligence_api.bootstrap import ConversationWorkerRuntime, WorkerRuntime
from speech_intelligence_api.domain.errors import (
    DependencyUnavailableError,
    DiarizationUnavailableError,
    ErrorCode,
    JobNotReadyError,
)
from speech_intelligence_api.ports.jobs import JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore
from speech_intelligence_api.workers import tasks
from tests.factories import make_settings


class FakeWorkerService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.executed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    async def execute(self, job_id: str) -> None:
        self.executed.append(job_id)
        if self.error is not None:
            raise self.error

    async def fail(self, job_id: str, failure_code: str) -> None:
        self.failed.append((job_id, failure_code))


class FakeBlobStore:
    def __init__(self) -> None:
        self.cutoff: datetime | None = None
        self.failure: Exception | None = None

    async def delete_expired(self, *, now: datetime | None = None) -> int:
        self.cutoff = now
        if self.failure is not None:
            raise self.failure
        return 1


class FakeJobStore:
    def __init__(self) -> None:
        self.cleaned = False

    async def cleanup_expired_index(self) -> int:
        self.cleaned = True
        return 1


class FakeRuntime:
    def __init__(self, service: FakeWorkerService) -> None:
        self.service = service
        self.blob_store = FakeBlobStore()
        self.job_store = FakeJobStore()
        self.cleanup_service = PrivacyCleanupService(
            blob_store=cast(EphemeralBlobStore, self.blob_store),
            job_store=cast(JobStore, self.job_store),
        )


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    service: FakeWorkerService,
    *,
    max_retries: int = 3,
) -> FakeRuntime:
    runtime = FakeRuntime(service)
    settings = make_settings().model_copy(update={"job_max_retries": max_retries})
    monkeypatch.setattr(tasks, "_runtime", lambda: cast(WorkerRuntime, runtime))
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    return runtime


def _install_conversation_runtime(
    monkeypatch: pytest.MonkeyPatch,
    service: FakeWorkerService,
    *,
    max_retries: int = 3,
) -> FakeRuntime:
    runtime = FakeRuntime(service)
    settings = make_settings().model_copy(update={"job_max_retries": max_retries})
    monkeypatch.setattr(
        tasks,
        "_conversation_runtime",
        lambda: cast(ConversationWorkerRuntime, runtime),
    )
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    return runtime


def test_process_task_success_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakeWorkerService()
    runtime = _install_runtime(monkeypatch, service)

    tasks.process_transcription.run("job_1")
    tasks.cleanup_expired_artifacts.run()

    assert service.executed == ["job_1"]
    assert service.failed == []
    assert runtime.blob_store.cutoff is not None
    assert runtime.job_store.cleaned is True


def test_cleanup_failure_is_sanitized_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, FakeWorkerService(), max_retries=0)
    runtime.blob_store.failure = OSError("private expired artifact path")

    with pytest.raises(DependencyUnavailableError) as captured:
        tasks.cleanup_expired_artifacts.run()

    assert "private expired artifact path" not in captured.value.public_message
    assert runtime.job_store.cleaned is True


def test_cleanup_transient_failure_requests_a_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, FakeWorkerService(), max_retries=1)
    runtime.blob_store.failure = OSError("private expired artifact path")
    retry_arguments: dict[str, object] = {}

    def request_retry(**kwargs: object) -> None:
        retry_arguments.update(kwargs)
        raise Retry("retry requested")

    monkeypatch.setattr(tasks.cleanup_expired_artifacts, "retry", request_retry)

    with pytest.raises(Retry) as captured:
        tasks.cleanup_expired_artifacts.run()

    assert "private expired artifact path" not in str(captured.value)
    assert isinstance(retry_arguments["exc"], DependencyUnavailableError)
    assert retry_arguments["max_retries"] == 1
    assert runtime.job_store.cleaned is True


def test_process_task_finalizes_exhausted_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(OSError("private path must not become job metadata"))
    _install_runtime(monkeypatch, service, max_retries=0)

    tasks.process_transcription.run("job_2")

    assert service.failed == [("job_2", ErrorCode.DEPENDENCY_UNAVAILABLE.value)]


def test_process_task_maps_expected_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(JobNotReadyError())
    _install_runtime(monkeypatch, service)

    tasks.process_transcription.run("job_3")

    assert service.failed == [("job_3", ErrorCode.CONFLICT.value)]


def test_process_task_marks_and_reraises_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(RuntimeError("unexpected"))
    _install_runtime(monkeypatch, service)

    with pytest.raises(RuntimeError, match="unexpected"):
        tasks.process_transcription.run("job_4")

    assert service.failed == [("job_4", ErrorCode.INTERNAL_ERROR.value)]


def test_runtime_refuses_to_start_when_async_jobs_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks._runtime.cache_clear()
    monkeypatch.setattr(tasks, "get_settings", make_settings)

    with pytest.raises(RuntimeError, match="disabled"):
        tasks._runtime()

    tasks._runtime.cache_clear()


def test_process_conversation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakeWorkerService()
    _install_conversation_runtime(monkeypatch, service)

    tasks.process_conversation.run("job_conversation")

    assert service.executed == ["job_conversation"]
    assert service.failed == []


def test_process_conversation_finalizes_exhausted_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(DiarizationUnavailableError())
    _install_conversation_runtime(monkeypatch, service, max_retries=0)

    tasks.process_conversation.run("job_model_failure")

    assert service.failed == [("job_model_failure", ErrorCode.DEPENDENCY_UNAVAILABLE.value)]


def test_process_conversation_maps_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(JobNotReadyError())
    _install_conversation_runtime(monkeypatch, service)

    tasks.process_conversation.run("job_conflict")

    assert service.failed == [("job_conflict", ErrorCode.CONFLICT.value)]


def test_process_conversation_marks_and_reraises_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeWorkerService(RuntimeError("unexpected conversation error"))
    _install_conversation_runtime(monkeypatch, service)

    with pytest.raises(RuntimeError, match="unexpected conversation"):
        tasks.process_conversation.run("job_unexpected")

    assert service.failed == [("job_unexpected", ErrorCode.INTERNAL_ERROR.value)]


def test_conversation_runtime_requires_both_feature_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks._conversation_runtime.cache_clear()
    monkeypatch.setattr(tasks, "get_settings", make_settings)

    with pytest.raises(RuntimeError, match="diarization jobs are disabled"):
        tasks._conversation_runtime()

    tasks._conversation_runtime.cache_clear()


class _FakeTaskRequest:
    def __init__(self, headers: dict[str, str], routing_key: object) -> None:
        self.headers = headers
        self.delivery_info = {"routing_key": routing_key}


class _FakeCeleryTask:
    def __init__(self, request: _FakeTaskRequest) -> None:
        self.request = request


def test_worker_observation_records_queue_wait_and_bounded_processing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings().model_copy(update={"metrics_enabled": True})
    observability = ServiceObservability(settings)
    monkeypatch.setattr(tasks, "_observability", lambda: observability)
    fake_task = cast(
        Task,
        cast(
            object,
            _FakeCeleryTask(
                _FakeTaskRequest(
                    {"speech-enqueued-at": str(time.time() - 2)},
                    "transcription.long",
                )
            ),
        ),
    )

    with tasks._observe_task(
        fake_task,
        kind="transcription",
        fallback_queue="transcription.short",
    ) as observation:
        observation.outcome = "completed"

    payload, _ = observability.render_metrics()
    text = payload.decode("utf-8")
    assert 'queue="transcription.long"' in text
    assert 'kind="transcription"' in text
    assert 'outcome="completed"' in text
    assert "speech_intelligence_job_queue_wait_seconds_count" in text
    assert tasks._enqueued_at({}) is None
    assert tasks._enqueued_at({"speech-enqueued-at": "invalid"}) is None

    invalid_queue_task = cast(
        Task,
        cast(object, _FakeCeleryTask(_FakeTaskRequest({}, object()))),
    )
    assert tasks._request_queue(invalid_queue_task, "diarization") == "diarization"
    asyncio.run(observability.close())
