"""Transcript export endpoint integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from speech_intelligence_api.entrypoints.http.app import create_app
from tests.factories import TEST_API_KEY, make_settings

_TRANSCRIPT = (
    "So yeah so yeah. The migraine started on Tuesday evening. Yeah so yeah. "
    "I took ibuprofen before bed. Yeah okay. So yeah so."
)


def _client() -> TestClient:
    settings = make_settings().model_copy(update={"rate_limit_enabled": False})
    return TestClient(create_app(settings))


def test_transcript_export_downloads_as_a_named_utf8_attachment() -> None:
    with _client() as client:
        response = client.post(
            "/v1/exports",
            headers={"X-API-Key": TEST_API_KEY},
            json={"transcript": _TRANSCRIPT, "content": "transcript", "format": "txt"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["content-disposition"].startswith("attachment; filename=")
    assert response.headers["cache-control"] == "no-store"
    assert "The migraine started on Tuesday evening." in response.content.decode("utf-8")


def test_summary_export_returns_only_the_distinctive_sentences() -> None:
    with _client() as client:
        response = client.post(
            "/v1/exports",
            headers={"X-API-Key": TEST_API_KEY},
            json={"transcript": _TRANSCRIPT, "content": "summary", "format": "txt"},
        )

    body = response.content.decode("utf-8")
    assert response.status_code == 200
    assert "The migraine started on Tuesday evening." in body
    assert "Yeah okay." not in body


def test_a_caller_supplied_title_names_the_download() -> None:
    with _client() as client:
        response = client.post(
            "/v1/exports",
            headers={"X-API-Key": TEST_API_KEY},
            json={"transcript": _TRANSCRIPT, "format": "txt", "title": "Visit Notes"},
        )

    assert 'filename="visit-notes.txt"' in response.headers["content-disposition"]


def test_export_requires_authentication() -> None:
    with _client() as client:
        response = client.post("/v1/exports", json={"transcript": _TRANSCRIPT})

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"


def test_a_blank_transcript_is_refused_without_echoing_it() -> None:
    with _client() as client:
        response = client.post(
            "/v1/exports",
            headers={"X-API-Key": TEST_API_KEY},
            json={"transcript": "   ", "format": "txt"},
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


def test_pdf_export_without_installed_fonts_points_the_caller_at_text() -> None:
    with _client() as client:
        response = client.post(
            "/v1/exports",
            headers={"X-API-Key": TEST_API_KEY},
            json={"transcript": _TRANSCRIPT, "format": "pdf"},
        )

    # No font directory is configured in tests, so PDF must fail loudly and
    # explain the supported alternative rather than returning a broken file.
    assert response.status_code == 422
    assert "txt" in response.json()["detail"]
