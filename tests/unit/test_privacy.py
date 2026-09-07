"""Private-data cleanup orchestration tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from speech_intelligence_api.application.privacy import PrivacyCleanupService
from speech_intelligence_api.domain.errors import DependencyUnavailableError
from speech_intelligence_api.ports.jobs import JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore


class CleanupBlobStore:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.cutoff: datetime | None = None

    async def delete_expired(self, *, now: datetime | None = None) -> int:
        self.cutoff = now
        if self.failure is not None:
            raise self.failure
        return 3


class CleanupJobStore:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.called = False

    async def cleanup_expired_index(self) -> int:
        self.called = True
        if self.failure is not None:
            raise self.failure
        return 2


@pytest.mark.asyncio
async def test_cleanup_reports_only_bounded_counts() -> None:
    blobs = CleanupBlobStore()
    jobs = CleanupJobStore()
    service = PrivacyCleanupService(
        blob_store=cast(EphemeralBlobStore, blobs),
        job_store=cast(JobStore, jobs),
    )
    now = datetime.now(tz=UTC)

    report = await service.execute(now=now)

    assert report.deleted_artifacts == 3
    assert report.removed_index_entries == 2
    assert blobs.cutoff == now
    assert jobs.called is True


@pytest.mark.asyncio
async def test_cleanup_attempts_both_layers_and_sanitizes_failure() -> None:
    private_message = "private storage path and Redis address"
    blobs = CleanupBlobStore(failure=OSError(private_message))
    jobs = CleanupJobStore(failure=ConnectionError(private_message))
    service = PrivacyCleanupService(
        blob_store=cast(EphemeralBlobStore, blobs),
        job_store=cast(JobStore, jobs),
    )

    with pytest.raises(DependencyUnavailableError) as captured:
        await service.execute()

    assert jobs.called is True
    assert private_message not in captured.value.public_message


@pytest.mark.asyncio
async def test_cleanup_supports_blob_only_direct_processing() -> None:
    blobs = CleanupBlobStore()
    service = PrivacyCleanupService(blob_store=cast(EphemeralBlobStore, blobs))

    report = await service.execute()

    assert report.deleted_artifacts == 3
    assert report.removed_index_entries == 0
