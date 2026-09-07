"""HTTP admission and queue-backpressure load contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
from fastapi.testclient import TestClient

from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionService,
    QueuedTranscriptionOutcome,
    UploadCommand,
)
from speech_intelligence_api.domain.enums import JobKind, JobStatus
from speech_intelligence_api.domain.errors import CapacityExceededError
from speech_intelligence_api.domain.models import JobRecord
from speech_intelligence_api.entrypoints.http.app import create_app
from speech_intelligence_api.entrypoints.load_test_client import LoadTestConfig, run_load
from tests.audio_fixtures import make_wav_bytes
from tests.factories import TEST_API_KEY, make_settings


class ConcurrentAdmissionService:
    """Fast service double that measures HTTP concurrency without model inference."""

    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0
        self.idempotency_keys: set[str] = set()

    async def execute(self, command: UploadCommand) -> QueuedTranscriptionOutcome:
        assert command.idempotency_key is not None
        self.idempotency_keys.add(command.idempotency_key)
        index = len(self.idempotency_keys)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0.001)
            now = datetime.now(tz=UTC)
            job = JobRecord(
                job_id=f"job_{index:032x}",
                kind=JobKind.TRANSCRIPTION,
                status=JobStatus.QUEUED,
                created_at=now,
                expires_at=now + timedelta(minutes=30),
            )
            return QueuedTranscriptionOutcome(job=job, replayed=False)
        finally:
            self.active -= 1


class CapacityService:
    async def execute(self, command: UploadCommand) -> QueuedTranscriptionOutcome:
        raise CapacityExceededError(1000, retry_after_seconds=7)


async def test_accepts_one_thousand_concurrent_http_job_submissions(tmp_path: Path) -> None:
    audio_path = tmp_path / "load.wav"
    audio_path.write_bytes(make_wav_bytes(duration_seconds=0.05))
    service = ConcurrentAdmissionService()
    settings = make_settings().model_copy(update={"log_level": "CRITICAL"})
    app = create_app(
        settings,
        transcription_service=cast(BatchTranscriptionService, service),
    )
    config = LoadTestConfig(
        audio_path=audio_path,
        api_key=TEST_API_KEY,
        submissions=1000,
        concurrency=1000,
        timeout_seconds=30,
    )

    summary = await run_load(
        config,
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
    )

    assert summary.passed
    assert summary.total == 1000
    assert summary.status_counts == (("202", 1000),)
    assert len(service.idempotency_keys) == 1000
    assert service.peak_active > 1


def test_full_queue_returns_explicit_retry_contract() -> None:
    app = create_app(
        make_settings(),
        transcription_service=cast(BatchTranscriptionService, CapacityService()),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/transcriptions",
            headers={
                "X-API-Key": TEST_API_KEY,
                "Idempotency-Key": "capacity-contract-0001",
            },
            files={"file": ("load.wav", make_wav_bytes(), "audio/wav")},
            data={
                "language_mode": "explicit",
                "language": "en",
                "processing_mode": "async",
            },
        )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    assert response.json()["code"] == "capacity_exceeded"
    assert response.json()["details"] == {
        "max_pending_jobs": 1000,
        "retry_after_seconds": 7,
    }
