"""Fail-closed private-artifact cleanup during API startup."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.domain.errors import DependencyUnavailableError
from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import make_settings


def test_api_deletes_overdue_artifacts_before_serving(tmp_path: Path) -> None:
    expired = tmp_path / "expired.upload"
    expired.write_bytes(b"private audio")
    old_time = (datetime.now(tz=UTC) - timedelta(minutes=31)).timestamp()
    os.utime(expired, (old_time, old_time))
    settings = make_settings(auth_enabled=False).model_copy(
        update={
            "temp_storage_root": tmp_path,
            "live_transcription_enabled": False,
        }
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert not expired.exists()


def test_api_refuses_startup_when_private_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_cleanup(
        self: LocalEphemeralBlobStore,
        *,
        now: datetime | None = None,
    ) -> int:
        raise OSError("private path must not escape")

    monkeypatch.setattr(LocalEphemeralBlobStore, "delete_expired", fail_cleanup)
    settings = make_settings(auth_enabled=False).model_copy(
        update={
            "temp_storage_root": tmp_path,
            "live_transcription_enabled": False,
        }
    )
    app = create_app(settings)

    with pytest.raises(DependencyUnavailableError) as captured, TestClient(app):
        pass

    assert "private path" not in captured.value.public_message
