"""Live PCM snapshot adapter tests."""

from __future__ import annotations

import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
import pytest

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.pcm_wave import PcmWaveSnapshotStore


def _directory_entries(path: Path) -> list[Path]:
    return list(path.iterdir())


@pytest.mark.asyncio
async def test_pcm_snapshot_is_private_wav_and_immediately_deletable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = LocalEphemeralBlobStore(tmp_path)
    snapshots = PcmWaveSnapshotStore(store)
    pcm = b"\x01\x00" * 512

    reference = await snapshots.create(
        pcm,
        sample_rate_hz=16_000,
        expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
    )

    path = store.resolve_path(reference)
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 16_000
        assert audio.readframes(512) == pcm
    assert await snapshots.delete(reference) is True
    assert await anyio.to_thread.run_sync(_directory_entries, tmp_path) == []


@pytest.mark.asyncio
async def test_pcm_snapshot_rejects_partial_samples(tmp_path) -> None:  # type: ignore[no-untyped-def]
    snapshots = PcmWaveSnapshotStore(LocalEphemeralBlobStore(tmp_path))

    with pytest.raises(ValueError, match="whole signed 16-bit"):
        await snapshots.create(
            b"x",
            sample_rate_hz=16_000,
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_pcm_snapshot_rejects_wrong_sample_rate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    snapshots = PcmWaveSnapshotStore(LocalEphemeralBlobStore(tmp_path))

    with pytest.raises(ValueError, match="16 kHz"):
        await snapshots.create(
            b"\0\0",
            sample_rate_hz=8_000,
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_pcm_snapshot_removes_partial_output_after_encoder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = PcmWaveSnapshotStore(LocalEphemeralBlobStore(tmp_path))

    def fail_after_write(destination: Path, pcm_audio: bytes, sample_rate_hz: int) -> None:
        assert pcm_audio == b"\0\0"
        assert sample_rate_hz == 16_000
        destination.write_bytes(b"partial")
        raise OSError("encoder failed")

    monkeypatch.setattr(snapshots, "_write_wave", fail_after_write)

    with pytest.raises(OSError, match="encoder failed"):
        await snapshots.create(
            b"\0\0",
            sample_rate_hz=16_000,
            expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )

    assert await anyio.to_thread.run_sync(_directory_entries, tmp_path) == []
