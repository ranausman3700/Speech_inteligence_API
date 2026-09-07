"""Secure local ephemeral storage tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore


async def _chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


def _directory_entries(path: Path) -> list[Path]:
    return list(path.iterdir())


def _resolved(path: Path) -> Path:
    return path.resolve()


@pytest.mark.asyncio
async def test_store_round_trip_and_idempotent_delete(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path, read_chunk_bytes=3)
    expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)

    reference = await store.put(
        _chunks(b"hello", b"-world"),
        media_type="audio/wav",
        expires_at=expires_at,
    )
    contents = b"".join([chunk async for chunk in store.read_chunks(reference)])

    assert contents == b"hello-world"
    assert reference.size_bytes == len(contents)
    assert reference.key.endswith(".upload")
    assert store.resolve_path(reference).parent == _resolved(tmp_path)
    assert await store.delete(reference) is True
    assert await store.delete(reference) is False


@pytest.mark.asyncio
async def test_store_deletes_partial_file_when_stream_fails(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)

    async def failing_stream() -> AsyncIterator[bytes]:
        yield b"partial"
        raise RuntimeError("failed upload")

    with pytest.raises(RuntimeError, match="failed upload"):
        await store.put(
            failing_stream(),
            media_type="audio/wav",
            expires_at=expires_at,
        )

    assert _directory_entries(tmp_path) == []


def test_store_rejects_unsafe_reservation_suffix(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)

    with pytest.raises(ValueError, match="simple extension"):
        store.reserve_path("../wav")


def test_store_rejects_reference_for_external_path(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path / "storage")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"data")

    with pytest.raises(ValueError, match="outside"):
        store.reference_for_path(
            outside,
            media_type="audio/wav",
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )


@pytest.mark.asyncio
async def test_store_cleanup_removes_only_expired_regular_files(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    expired = store.reserve_path(".wav")
    current = store.reserve_path(".wav")
    expired.write_bytes(b"old")
    current.write_bytes(b"new")
    old_time = (datetime.now(tz=UTC) - timedelta(minutes=31)).timestamp()
    os.utime(expired, (old_time, old_time))

    deleted = await store.delete_expired(now=datetime.now(tz=UTC) - timedelta(minutes=30))

    assert deleted == 1
    assert not expired.exists()
    assert current.exists()


def test_store_fails_closed_when_expiry_marker_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    reserved = store.reserve_path(".wav")
    reserved.write_bytes(b"private")

    def fail_expiry_marker(*_: object) -> None:
        raise PermissionError("private filesystem detail")

    monkeypatch.setattr(os, "utime", fail_expiry_marker)

    with pytest.raises(PermissionError, match="private filesystem detail"):
        store.reference_for_path(
            reserved,
            media_type="audio/wav",
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )

    assert not reserved.exists()


@pytest.mark.asyncio
async def test_cleanup_continues_after_one_artifact_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    blocked = store.reserve_path(".wav")
    removable = store.reserve_path(".wav")
    old_time = (datetime.now(tz=UTC) - timedelta(minutes=31)).timestamp()
    os.utime(blocked, (old_time, old_time))
    os.utime(removable, (old_time, old_time))
    original_delete = store._delete_path

    def delete_with_one_failure(path: Path) -> bool:
        if path == blocked:
            raise PermissionError("private blocked path")
        return original_delete(path)

    monkeypatch.setattr(store, "_delete_path", delete_with_one_failure)

    with pytest.raises(OSError, match="cleanup was incomplete"):
        await store.delete_expired()

    assert blocked.exists()
    assert not removable.exists()
    assert blocked.name not in caplog.text
    assert any(
        getattr(record, "exception_class", None) == "PermissionError" for record in caplog.records
    )
