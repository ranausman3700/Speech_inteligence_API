"""Unit contracts for the privacy-safe load client."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from speech_intelligence_api.entrypoints import load_test_client
from speech_intelligence_api.entrypoints.load_test_client import (
    LoadSummary,
    LoadTestConfig,
    run_load,
)


def _audio(tmp_path: Path, *, name: str = "private-customer-name.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"RIFF" + b"\x00" * 1024)
    return path


def test_load_config_rejects_unsafe_or_inconsistent_parameters(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    with pytest.raises(ValueError, match="without credentials"):
        LoadTestConfig(audio_path=audio, base_url="http://secret@example.test")
    with pytest.raises(ValueError, match="cannot include"):
        LoadTestConfig(audio_path=audio, language_mode="automatic", language="en")
    with pytest.raises(ValueError, match="submission count"):
        LoadTestConfig(audio_path=audio, submissions=2, concurrency=3)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"submissions": 0}, "submissions"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"max_audio_bytes": 100}, "maximum audio bytes"),
        ({"api_key": "   "}, "API key"),
        ({"language_mode": "unsupported"}, "language mode"),
        ({"language": None}, "requires a language"),
        ({"allowed_statuses": frozenset()}, "allowed statuses"),
    ],
)
def test_load_config_rejects_invalid_bounds(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {"audio_path": _audio(tmp_path)}
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        LoadTestConfig(**values)  # type: ignore[arg-type]


async def test_load_summary_contains_only_bounded_operational_data(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    api_key = "private-api-key-value"
    observed_keys: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_keys.add(request.headers["Idempotency-Key"])
        assert request.headers["X-API-Key"] == api_key
        body = await request.aread()
        assert b"load-audio.wav" in body
        assert audio.name.encode() not in body
        return httpx.Response(
            503,
            json={
                "code": "capacity_exceeded",
                "detail": "private upstream diagnostic that must not be reported",
            },
            headers={"Retry-After": "5"},
        )

    config = LoadTestConfig(
        audio_path=audio,
        api_key=api_key,
        submissions=4,
        concurrency=4,
        allowed_statuses=frozenset({503}),
    )
    summary = await run_load(config, transport=httpx.MockTransport(handler))
    rendered = summary.to_json()

    assert summary.passed
    assert summary.status_counts == (("503", 4),)
    assert summary.error_counts == (("capacity_exceeded", 4),)
    assert len(observed_keys) == 4
    assert api_key not in repr(config)
    assert api_key not in rendered
    assert audio.name not in rendered
    assert "private upstream diagnostic" not in rendered


async def test_load_client_sanitizes_transport_failures(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private network topology", request=request)

    config = LoadTestConfig(audio_path=audio, submissions=2, concurrency=1)
    summary = await run_load(config, transport=httpx.MockTransport(handler))

    assert not summary.passed
    assert summary.failed == 2
    assert summary.status_counts == (("transport", 2),)
    assert summary.error_counts == (("transport_error", 2),)
    assert "private network topology" not in summary.to_json()


async def test_load_client_sanitizes_timeout_and_local_file_failures(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private timeout detail", request=request)

    config = LoadTestConfig(audio_path=audio, submissions=1, concurrency=1)
    timeout_summary = await run_load(config, transport=httpx.MockTransport(timeout_handler))
    audio.unlink()
    missing_file_summary = await run_load(
        config,
        transport=httpx.MockTransport(lambda request: httpx.Response(202)),
    )

    assert timeout_summary.error_counts == (("timeout", 1),)
    assert missing_file_summary.error_counts == (("local_file_error", 1),)
    assert "private timeout detail" not in timeout_summary.to_json()


async def test_load_client_bounds_unexpected_error_responses(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    responses = iter(
        (
            httpx.Response(503, content=b"not-json"),
            httpx.Response(503, json={"code": "private-unbounded-code"}),
        )
    )
    config = LoadTestConfig(audio_path=audio, submissions=2, concurrency=1)

    summary = await run_load(
        config,
        transport=httpx.MockTransport(lambda request: next(responses)),
    )

    assert summary.error_counts == (("unexpected_response", 2),)


def test_main_reads_api_key_from_environment_and_prints_only_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = _audio(tmp_path)
    secret = "private-main-api-key"
    observed: list[LoadTestConfig] = []

    async def fake_run(config: LoadTestConfig) -> LoadSummary:
        observed.append(config)
        return LoadSummary(
            total=1,
            accepted=1,
            failed=0,
            elapsed_seconds=0.1,
            requests_per_second=10,
            latency_p50_ms=1,
            latency_p95_ms=1,
            latency_p99_ms=1,
            status_counts=(("202", 1),),
            error_counts=(),
        )

    monkeypatch.setenv("SPEECH_LOAD_TEST_API_KEY", secret)
    monkeypatch.setattr(load_test_client, "run_load", fake_run)

    exit_code = load_test_client.main(
        ["--audio", str(audio), "--submissions", "1", "--concurrency", "1"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert observed[0].api_key == secret
    assert secret not in output
    assert audio.name not in output
    assert '"passed": true' in output


def test_main_requires_an_explicit_authentication_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_LOAD_TEST_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="2"):
        load_test_client.main(["--audio", str(_audio(tmp_path))])
