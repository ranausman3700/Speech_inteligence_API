"""Ephemeral job-state and broker boundaries."""

from typing import Protocol

from speech_intelligence_api.domain.enums import JobStatus
from speech_intelligence_api.domain.jobs import (
    JobPayload,
    JobReservation,
    JobResult,
)
from speech_intelligence_api.domain.models import JobRecord


class JobStore(Protocol):
    """Persist temporary job state/results without extending absolute expiry."""

    async def reserve(
        self,
        job: JobRecord,
        payload: JobPayload,
        *,
        max_pending_jobs: int,
        idempotency_digest: str | None = None,
        request_fingerprint: str | None = None,
    ) -> JobReservation:
        """Atomically enforce capacity, job uniqueness, and idempotency."""

    async def get(self, job_id: str) -> JobRecord | None:
        """Return the unexpired job or ``None``."""

    async def get_payload(self, job_id: str) -> JobPayload | None:
        """Return private worker input for an unexpired job."""

    async def attach_task(self, job_id: str, task_id: str) -> JobRecord | None:
        """Associate the broker task ID with a live job."""

    async def transition(
        self,
        job_id: str,
        target: JobStatus,
        *,
        progress_percent: int | None = None,
        failure_code: str | None = None,
    ) -> JobRecord | None:
        """Atomically apply a legal lifecycle transition."""

    async def update_progress(self, job_id: str, progress_percent: int) -> JobRecord | None:
        """Advance progress without changing lifecycle state."""

    async def request_cancellation(self, job_id: str) -> JobRecord | None:
        """Atomically mark a live job for cooperative cancellation."""

    async def save_result(
        self,
        job_id: str,
        result: JobResult,
    ) -> JobRecord | None:
        """Atomically store a result and mark the job successful."""

    async def get_result(self, job_id: str) -> JobResult | None:
        """Return the result while its parent job remains live."""

    async def delete(self, job_id: str) -> bool:
        """Delete all job state/results and return whether state existed."""

    async def ping(self) -> None:
        """Raise if the backing state service is unavailable."""

    async def cleanup_expired_index(self) -> int:
        """Remove expired non-payload capacity bookkeeping."""


class JobDispatcher(Protocol):
    """Enqueue and revoke opaque background tasks."""

    async def enqueue(self, job_id: str, *, queue: str, priority: int) -> str:
        """Enqueue a job and return the broker task ID."""

    async def revoke(self, task_id: str) -> None:
        """Prevent a queued task from starting when possible."""
