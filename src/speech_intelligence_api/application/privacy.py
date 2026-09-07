"""Fail-closed cleanup of private artifacts and ephemeral job bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from speech_intelligence_api.domain.errors import DependencyUnavailableError
from speech_intelligence_api.ports.jobs import JobStore
from speech_intelligence_api.ports.storage import EphemeralBlobStore


@dataclass(frozen=True, slots=True)
class PrivacyCleanupReport:
    """Bounded operational counts containing no private identifiers."""

    deleted_artifacts: int
    removed_index_entries: int


class PrivacyCleanupService:
    """Attempt independent cleanup layers and sanitize all dependency failures."""

    def __init__(
        self,
        *,
        blob_store: EphemeralBlobStore,
        job_store: JobStore | None = None,
    ) -> None:
        self._blob_store = blob_store
        self._job_store = job_store

    async def execute(self, *, now: datetime | None = None) -> PrivacyCleanupReport:
        """Delete expired data without allowing one failed layer to skip the other."""

        cutoff = now or datetime.now(tz=UTC)
        failed = False
        deleted_artifacts = 0
        removed_index_entries = 0
        try:
            deleted_artifacts = await self._blob_store.delete_expired(now=cutoff)
        except Exception:
            failed = True
        if self._job_store is not None:
            try:
                removed_index_entries = await self._job_store.cleanup_expired_index()
            except Exception:
                failed = True
        if failed:
            raise DependencyUnavailableError from None
        return PrivacyCleanupReport(
            deleted_artifacts=deleted_artifacts,
            removed_index_entries=removed_index_entries,
        )
