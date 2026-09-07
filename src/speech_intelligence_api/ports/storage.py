"""Private ephemeral object-storage boundary."""

from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
from typing import Protocol

from speech_intelligence_api.domain.models import BlobReference


class EphemeralBlobStore(Protocol):
    """Store private artifacts without permitting expiry extension."""

    async def put(
        self,
        stream: AsyncIterable[bytes],
        *,
        media_type: str,
        expires_at: datetime,
    ) -> BlobReference:
        """Write a bounded stream and return an opaque reference."""

    def read_chunks(self, reference: BlobReference) -> AsyncIterator[bytes]:
        """Yield private object bytes without exposing its physical location."""

    async def delete(self, reference: BlobReference) -> bool:
        """Permanently delete the object and report whether it existed."""

    async def delete_expired(self, *, now: datetime | None = None) -> int:
        """Delete expired objects left by interrupted processing."""
