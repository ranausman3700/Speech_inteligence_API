"""Celery configuration and dispatch boundary tests."""

from unittest.mock import ANY, MagicMock

import pytest

from speech_intelligence_api.adapters.celery_jobs import CeleryJobDispatcher
from speech_intelligence_api.adapters.observability import ServiceObservability
from speech_intelligence_api.domain.enums import JobQueue
from speech_intelligence_api.workers.celery_app import (
    CLEANUP_TASK_NAME,
    TRANSCRIPTION_TASK_NAME,
    create_celery_app,
)
from tests.factories import make_settings


def test_celery_configuration_is_private_bounded_and_queue_isolated() -> None:
    settings = make_settings()
    app = create_celery_app(settings)

    assert app.conf.accept_content == ["json"]
    assert app.conf.task_serializer == "json"
    assert app.conf.task_ignore_result is True
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_soft_time_limit < app.conf.task_time_limit
    assert {queue.name for queue in app.conf.task_queues} == {queue.value for queue in JobQueue}
    assert all(queue.exchange.name == queue.name for queue in app.conf.task_queues)
    assert all(queue.routing_key == queue.name for queue in app.conf.task_queues)
    assert app.conf.task_routes[TRANSCRIPTION_TASK_NAME]["queue"] == "transcription.short"
    schedule = app.conf.beat_schedule["cleanup-expired-private-artifacts"]
    assert schedule["task"] == CLEANUP_TASK_NAME


@pytest.mark.asyncio
async def test_dispatcher_enqueues_only_opaque_id_and_revokes_without_terminate() -> None:
    app = MagicMock()
    result = MagicMock()
    result.id = "task-123"
    app.send_task.return_value = result
    dispatcher = CeleryJobDispatcher(app)

    task_id = await dispatcher.enqueue(
        "job_123",
        queue=JobQueue.LONG_AUDIO.value,
        priority=7,
    )
    await dispatcher.revoke(task_id)

    assert task_id == "task-123"
    app.send_task.assert_called_once_with(
        TRANSCRIPTION_TASK_NAME,
        args=["job_123"],
        queue=JobQueue.LONG_AUDIO.value,
        priority=7,
        headers=ANY,
    )
    headers = app.send_task.call_args.kwargs["headers"]
    assert set(headers) == {"speech-enqueued-at"}
    assert float(headers["speech-enqueued-at"]) > 0
    app.control.revoke.assert_called_once_with("task-123", terminate=False)


@pytest.mark.asyncio
async def test_dispatcher_rejects_invalid_priority() -> None:
    dispatcher = CeleryJobDispatcher(MagicMock())

    with pytest.raises(ValueError, match="between 0 and 9"):
        await dispatcher.enqueue("job_123", queue="transcription.short", priority=10)


@pytest.mark.asyncio
async def test_dispatcher_records_sanitized_publication_failure() -> None:
    app = MagicMock()
    app.send_task.side_effect = RuntimeError("private broker details")
    settings = make_settings().model_copy(update={"metrics_enabled": True})
    observability = ServiceObservability(settings)
    dispatcher = CeleryJobDispatcher(
        app,
        observability=observability,
        clock=lambda: 100.0,
    )

    with pytest.raises(RuntimeError, match="private broker details"):
        await dispatcher.enqueue(
            "job_private-value-must-not-be-telemetry",
            queue=JobQueue.SHORT_TRANSCRIPTION.value,
            priority=5,
        )

    payload, _ = observability.render_metrics()
    text = payload.decode("utf-8")
    assert 'outcome="rejected",queue="transcription.short"' in text
    assert "private broker details" not in text
    assert "job_private" not in text
    await observability.close()
