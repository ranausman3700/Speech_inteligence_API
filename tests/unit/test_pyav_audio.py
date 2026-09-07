"""Actual PyAV probing and normalization tests."""

from __future__ import annotations

import wave
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.pyav_audio import PyAvAudioPreprocessor
from speech_intelligence_api.domain.errors import AudioTooLongError, InvalidAudioError
from speech_intelligence_api.domain.models import BlobReference
from tests.audio_fixtures import make_empty_wav_bytes, make_wav_bytes


async def _one_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


async def _stored_audio(
    store: LocalEphemeralBlobStore,
    contents: bytes,
) -> tuple[BlobReference, datetime]:
    expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)
    reference = await store.put(
        _one_chunk(contents),
        media_type="audio/wav",
        expires_at=expires_at,
    )
    return reference, expires_at


def _directory_entries(path: Path) -> list[Path]:
    return list(path.iterdir())


@pytest.mark.asyncio
async def test_probe_and_normalize_real_wav(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    preprocessor = PyAvAudioPreprocessor(store)
    reference, expires_at = await _stored_audio(
        store,
        make_wav_bytes(duration_seconds=0.2, sample_rate_hz=8_000, channels=2),
    )

    probe = await preprocessor.probe(reference)
    normalized, duration = await preprocessor.normalize(
        reference,
        expires_at=expires_at,
        max_duration_seconds=1,
    )

    assert probe.detected_media_type == "audio/wav"
    assert probe.codec == "pcm_s16le"
    assert probe.channels == 2
    assert probe.sample_rate_hz == 8_000
    assert probe.duration_seconds == pytest.approx(0.2, abs=0.01)
    assert duration == pytest.approx(0.2, abs=0.01)
    with wave.open(str(store.resolve_path(normalized)), "rb") as output:
        assert output.getnchannels() == 1
        assert output.getframerate() == 16_000
        assert output.getsampwidth() == 2


@pytest.mark.asyncio
async def test_probe_rejects_non_media_bytes(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    preprocessor = PyAvAudioPreprocessor(store)
    reference, _ = await _stored_audio(store, b"not-media")

    with pytest.raises(InvalidAudioError):
        await preprocessor.probe(reference)


@pytest.mark.asyncio
async def test_normalization_enforces_decoded_duration_limit(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    preprocessor = PyAvAudioPreprocessor(store)
    reference, expires_at = await _stored_audio(
        store,
        make_wav_bytes(duration_seconds=0.2),
    )

    with pytest.raises(AudioTooLongError):
        await preprocessor.normalize(
            reference,
            expires_at=expires_at,
            max_duration_seconds=0.05,
        )

    assert len(_directory_entries(tmp_path)) == 1


@pytest.mark.asyncio
async def test_normalization_rejects_audio_without_samples(tmp_path: Path) -> None:
    store = LocalEphemeralBlobStore(tmp_path)
    preprocessor = PyAvAudioPreprocessor(store)
    reference, expires_at = await _stored_audio(store, make_empty_wav_bytes())

    with pytest.raises(InvalidAudioError) as captured:
        await preprocessor.normalize(
            reference,
            expires_at=expires_at,
            max_duration_seconds=1,
        )
    assert "no decodable samples" in captured.value.public_message


@pytest.mark.parametrize(
    ("format_name", "media_type"),
    [
        ("mp3", "audio/mpeg"),
        ("flac", "audio/flac"),
        ("ogg", "audio/ogg"),
        ("matroska,webm", "audio/webm"),
        ("mov,mp4,m4a,3gp,3g2,mj2", "audio/mp4"),
        ("aac", "audio/aac"),
    ],
)
def test_container_format_mapping(format_name: str, media_type: str) -> None:
    assert PyAvAudioPreprocessor._media_type_for_format(format_name) == media_type


def test_unknown_container_and_codec_are_rejected() -> None:
    with pytest.raises(InvalidAudioError) as container_error:
        PyAvAudioPreprocessor._media_type_for_format("unknown")
    assert "container" in container_error.value.public_message
    with pytest.raises(InvalidAudioError) as codec_error:
        PyAvAudioPreprocessor._validate_codec("dangerous")
    assert "codec" in codec_error.value.public_message
    PyAvAudioPreprocessor._validate_codec("adpcm_ima_wav")
