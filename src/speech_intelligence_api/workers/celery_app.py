"""Celery application with isolated workload queues and bounded delivery."""

from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from speech_intelligence_api.config import Settings, get_settings
from speech_intelligence_api.domain.enums import JobQueue

TRANSCRIPTION_TASK_NAME = "speech_intelligence.process_transcription"
CONVERSATION_TASK_NAME = "speech_intelligence.process_conversation"
CLEANUP_TASK_NAME = "speech_intelligence.cleanup_expired_artifacts"


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Create the worker application without loading inference models."""

    resolved = settings or get_settings()
    app = Celery(
        resolved.service_name,
        broker=resolved.celery_broker_url.get_secret_value(),
        include=["speech_intelligence_api.workers.tasks"],
    )
    queue_names = tuple(queue.value for queue in JobQueue)
    app.conf.update(
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_track_started=False,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_queue=JobQueue.SHORT_TRANSCRIPTION.value,
        task_default_priority=5,
        task_queue_max_priority=9,
        task_queues=tuple(
            Queue(
                name,
                exchange=Exchange(name, type="direct", durable=True),
                routing_key=name,
                durable=True,
                queue_arguments={"x-max-priority": 9},
            )
            for name in queue_names
        ),
        task_routes={
            TRANSCRIPTION_TASK_NAME: {
                "queue": JobQueue.SHORT_TRANSCRIPTION.value,
            },
            CONVERSATION_TASK_NAME: {
                "queue": JobQueue.DIARIZATION.value,
            },
            CLEANUP_TASK_NAME: {
                "queue": JobQueue.CLEANUP.value,
            },
        },
        task_soft_time_limit=resolved.job_soft_time_limit_seconds,
        task_time_limit=resolved.job_hard_time_limit_seconds,
        worker_prefetch_multiplier=resolved.celery_worker_prefetch_multiplier,
        worker_max_tasks_per_child=25,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": resolved.celery_visibility_timeout_seconds,
            "queue_order_strategy": "priority",
            "priority_steps": list(range(10)),
        },
        beat_schedule={
            "cleanup-expired-private-artifacts": {
                "task": CLEANUP_TASK_NAME,
                "schedule": resolved.artifact_cleanup_interval_seconds,
                "options": {"queue": JobQueue.CLEANUP.value, "priority": 0},
            }
        },
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()
