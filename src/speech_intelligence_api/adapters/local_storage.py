"""Secure local implementation of the ephemeral blob-store boundary."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio

from speech_intelligence_api.domain.models import BlobReference

logger = logging.getLogger(__name__)


class LocalEphemeralBlobStore:
    """Store randomized private files beneath one non-public directory."""

    def __init__(self, root: Path, *, read_chunk_bytes: int = 1024 * 1024) -> None:
        self._root = root.expanduser().resolve()
        self._read_chunk_bytes = read_chunk_bytes
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with suppress(OSError):
            self._root.chmod(0o700)

    @property
    def root(self) -> Path:
        """Return the configured root for adapter composition and diagnostics."""

        return self._root

    async def put(
        self,
        stream: AsyncIterable[bytes],
        *,
        media_type: str,
        expires_at: datetime,
    ) -> BlobReference:
        """Write a stream to a randomized file and remove partial data on failure."""

        path = self.reserve_path(".upload")
        size_bytes = 0
        try:
            async with await anyio.open_file(path, "wb") as destination:
                async for chunk in stream:
                    size_bytes += await destination.write(chunk)
            with suppress(OSError):
                await anyio.to_thread.run_sync(path.chmod, 0o600)
            return self.reference_for_path(
                path,
                media_type=media_type,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )
        except BaseException:
            await anyio.to_thread.run_sync(self._delete_path, path)
            raise

    async def read_chunks(self, reference: BlobReference) -> AsyncIterator[bytes]:
        """Yield object bytes from the resolved private path."""

        path = self.resolve_path(reference)
        async with await anyio.open_file(path, "rb") as source:
            while chunk := await source.read(self._read_chunk_bytes):
                yield chunk

    async def delete(self, reference: BlobReference) -> bool:
        """Idempotently delete one resolved private object."""

        return await anyio.to_thread.run_sync(self._delete_path, self.resolve_path(reference))

    async def delete_expired(self, *, now: datetime | None = None) -> int:
        """Delete stale artifacts left behind by interrupted processes."""

        cutoff = (now or datetime.now(tz=UTC)).timestamp()
        return await anyio.to_thread.run_sync(self._delete_paths_older_than, cutoff)

    def reserve_path(self, suffix: str) -> Path:
        """Atomically reserve a randomized file for a co-located adapter."""

        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ValueError("storage suffix must be a simple extension")
        for _ in range(10):
            path = self._root / f"{uuid4().hex}{suffix}"
            try:
                file_descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                continue
            os.close(file_descriptor)
            return path
        raise RuntimeError("could not reserve a unique ephemeral object")

    def reference_for_path(
        self,
        path: Path,
        *,
        media_type: str,
        expires_at: datetime,
        size_bytes: int | None = None,
    ) -> BlobReference:
        """Build an opaque reference for an adapter-created object."""

        resolved_path = path.resolve(strict=True)
        if resolved_path.parent != self._root:
            raise ValueError("ephemeral object is outside the configured storage root")
        actual_size = resolved_path.stat().st_size if size_bytes is None else size_bytes
        created_at = datetime.now(tz=UTC)
        try:
            reference = BlobReference(
                key=resolved_path.name,
                media_type=media_type,
                size_bytes=actual_size,
                created_at=created_at,
                expires_at=expires_at,
            )
            expiry_timestamp = expires_at.timestamp()
            os.utime(resolved_path, (expiry_timestamp, expiry_timestamp))
            return reference
        except BaseException:
            self._delete_path(resolved_path)
            raise

    def resolve_path(self, reference: BlobReference) -> Path:
        """Resolve an opaque key while preventing traversal and symlink escape."""

        candidate = (self._root / reference.key).resolve(strict=False)
        if candidate.parent != self._root:
            raise ValueError("ephemeral object resolved outside the configured storage root")
        return candidate

    @staticmethod
    def _delete_path(path: Path) -> bool:
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def _delete_paths_older_than(self, cutoff: float) -> int:
        deleted = 0
        deletion_failed = False
        for path in self._root.iterdir():
            try:
                if path.is_symlink() or not path.is_file() or path.stat().st_mtime > cutoff:
                    continue
                resolved = path.resolve(strict=True)
                if resolved.parent != self._root:
                    continue
                if self._delete_path(resolved):
                    deleted += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                deletion_failed = True
                logger.warning(
                    "Expired private artifact could not be deleted",
                    extra={"exception_class": type(exc).__name__},
                )
        if deletion_failed:
            raise OSError("expired private artifact cleanup was incomplete")
        return deleted
