"""Shared pytest fixtures."""

import os
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from speech_intelligence_api.config import Settings
from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import TEST_API_KEY, make_settings


@pytest.fixture(autouse=True)
def isolate_application_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a developer's local configuration from changing test outcomes."""

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in tuple(os.environ):
        if name.upper().startswith("SPEECH_API_"):
            monkeypatch.delenv(name)


@pytest.fixture
def settings() -> Settings:
    """Return secure test settings."""

    return make_settings()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """Return the application with production adapters composed."""

    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Run the application lifespan for each HTTP test."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return valid API authentication headers."""

    return {"X-API-Key": TEST_API_KEY}
