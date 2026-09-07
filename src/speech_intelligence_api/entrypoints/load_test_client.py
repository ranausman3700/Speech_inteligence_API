"""Privacy-safe asynchronous job-admission load client."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import secrets
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Final
from urllib.parse import urlsplit

import httpx

from speech_intelligence_api.application.audio_uploads import DECLARED_MEDIA_BY_EXTENSION
from speech_intelligence_api.domain.errors import ErrorCode

_DEFAULT_API_KEY_ENV: Final = "SPEECH_LOAD_TEST_API_KEY"
_MAX_AUDIO_BYTES: Final = 5 * 1024 * 1024
_KNOWN_ERROR_CODES: Final = frozenset(code.value for code in ErrorCode)


@dataclass(frozen=True, slots=True)
class LoadTestConfig:
    """Validated parameters for one bounded admission test."""

    audio_path: Path
    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = field(default=None, repr=False)
    submissions: int = 1000
    concurrency: int = 100
    timeout_seconds: float = 30.0
    max_audio_bytes: int = _MAX_AUDIO_BYTES
    language_mode: str = "explicit"
    language: str | None = "en"
    allowed_statuses: frozenset[int] = frozenset({202})

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base URL must be an explicit HTTP(S) URL without credentials")
        if not self.audio_path.is_file():
            raise ValueError("audio path must reference an existing file")
        extension = self.audio_path.suffix.casefold()
        if extension not in DECLARED_MEDIA_BY_EXTENSION:
            raise ValueError("audio file extension is not supported by this service")
        if not 1024 <= self.max_audio_bytes <= 100 * 1024 * 1024:
            raise ValueError("maximum audio bytes must be between 1 KiB and 100 MiB")
        audio_bytes = self.audio_path.stat().st_size
        if audio_bytes <= 0 or audio_bytes > self.max_audio_bytes:
            raise ValueError("audio file must be non-empty and within the load-client byte limit")
        if not 1 <= self.submissions <= 100_000:
            raise ValueError("submissions must be between 1 and 100000")
        if not 1 <= self.concurrency <= min(self.submissions, 2000):
            raise ValueError("concurrency must be between 1 and the submission count")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("timeout must be between 1 and 300 seconds")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("API key cannot be empty")
        if self.language_mode not in {"automatic", "explicit"}:
            raise ValueError("language mode must be automatic or explicit")
        if self.language_mode == "explicit" and not self.language:
            raise ValueError("explicit language mode requires a language code")
        if self.language_mode == "automatic" and self.language is not None:
            raise ValueError("automatic language mode cannot include a language code")
        if not self.allowed_statuses or any(
            status < 100 or status > 599 for status in self.allowed_statuses
        ):
            raise ValueError("allowed statuses must contain valid HTTP status codes")


@dataclass(frozen=True, slots=True)
class LoadSample:
    """One sanitized request outcome."""

    status_code: int | None
    latency_seconds: float
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class LoadSummary:
    """Bounded aggregate output that never contains uploaded or response content."""

    total: int
    accepted: int
    failed: int
    elapsed_seconds: float
    requests_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    status_counts: tuple[tuple[str, int], ...]
    error_counts: tuple[tuple[str, int], ...]

    @property
    def passed(self) -> bool:
        """Return whether every outcome matched the operator-approved contract."""

        return self.failed == 0

    def to_json(self) -> str:
        """Serialize only bounded aggregate fields."""

        return json.dumps(
            {
                "total": self.total,
                "accepted": self.accepted,
                "failed": self.failed,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "requests_per_second": round(self.requests_per_second, 2),
                "latency_ms": {
                    "p50": round(self.latency_p50_ms, 2),
                    "p95": round(self.latency_p95_ms, 2),
                    "p99": round(self.latency_p99_ms, 2),
                },
                "status_counts": dict(self.status_counts),
                "error_counts": dict(self.error_counts),
                "passed": self.passed,
            },
            sort_keys=True,
        )


async def run_load(
    config: LoadTestConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LoadSummary:
    """Submit bounded asynchronous jobs and return privacy-safe aggregate results."""

    run_id = secrets.token_hex(8)
    extension = config.audio_path.suffix.casefold()
    media_type = sorted(DECLARED_MEDIA_BY_EXTENSION[extension])[0]
    endpoint = f"{config.base_url.rstrip('/')}/v1/transcriptions"
    limits = httpx.Limits(
        max_connections=config.concurrency,
        max_keepalive_connections=config.concurrency,
    )
    timeout = httpx.Timeout(config.timeout_seconds)
    started_at = perf_counter()
    async with httpx.AsyncClient(transport=transport, limits=limits, timeout=timeout) as client:
        indices = iter(range(config.submissions))

        async def worker() -> list[LoadSample]:
            worker_samples: list[LoadSample] = []
            for index in indices:
                worker_samples.append(
                    await _submit_one(
                        client,
                        endpoint=endpoint,
                        config=config,
                        media_type=media_type,
                        extension=extension,
                        idempotency_key=f"load-{run_id}-{index:08d}",
                    )
                )
            return worker_samples

        worker_results = await asyncio.gather(
            *(asyncio.create_task(worker()) for _ in range(config.concurrency))
        )
        samples = [sample for worker_samples in worker_results for sample in worker_samples]
    elapsed_seconds = max(perf_counter() - started_at, sys.float_info.epsilon)
    return _summarize(samples, config.allowed_statuses, elapsed_seconds)


async def _submit_one(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    config: LoadTestConfig,
    media_type: str,
    extension: str,
    idempotency_key: str,
) -> LoadSample:
    headers = {"Idempotency-Key": idempotency_key}
    if config.api_key is not None:
        headers["X-API-Key"] = config.api_key
    form = {
        "language_mode": config.language_mode,
        "processing_mode": "async",
        "word_timestamps": "true",
    }
    if config.language is not None:
        form["language"] = config.language

    started_at = perf_counter()
    try:
        with config.audio_path.open("rb") as audio_stream:
            response = await client.post(
                endpoint,
                headers=headers,
                files={"file": (f"load-audio{extension}", audio_stream, media_type)},
                data=form,
            )
        return LoadSample(
            status_code=response.status_code,
            latency_seconds=perf_counter() - started_at,
            error_code=_safe_error_code(response),
        )
    except httpx.TimeoutException:
        return LoadSample(
            status_code=None,
            latency_seconds=perf_counter() - started_at,
            error_code="timeout",
        )
    except httpx.RequestError:
        return LoadSample(
            status_code=None,
            latency_seconds=perf_counter() - started_at,
            error_code="transport_error",
        )
    except OSError:
        return LoadSample(
            status_code=None,
            latency_seconds=perf_counter() - started_at,
            error_code="local_file_error",
        )


def _safe_error_code(response: httpx.Response) -> str | None:
    if response.status_code < 400:
        return None
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "unexpected_response"
    code = body.get("code") if isinstance(body, dict) else None
    return code if isinstance(code, str) and code in _KNOWN_ERROR_CODES else "unexpected_response"


def _summarize(
    samples: Sequence[LoadSample],
    allowed_statuses: frozenset[int],
    elapsed_seconds: float,
) -> LoadSummary:
    accepted = sum(sample.status_code in allowed_statuses for sample in samples)
    status_counts = Counter(
        str(sample.status_code) if sample.status_code is not None else "transport"
        for sample in samples
    )
    error_counts = Counter(sample.error_code for sample in samples if sample.error_code is not None)
    latencies_ms = sorted(sample.latency_seconds * 1000 for sample in samples)
    return LoadSummary(
        total=len(samples),
        accepted=accepted,
        failed=len(samples) - accepted,
        elapsed_seconds=elapsed_seconds,
        requests_per_second=len(samples) / elapsed_seconds,
        latency_p50_ms=_percentile(latencies_ms, 50),
        latency_p95_ms=_percentile(latencies_ms, 95),
        latency_p99_ms=_percentile(latencies_ms, 99),
        status_counts=tuple(sorted(status_counts.items())),
        error_counts=tuple(sorted(error_counts.items())),
    )


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil((percentile / 100) * len(values)) - 1)
    return values[index]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a bounded asynchronous transcription admission load test.",
    )
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--submissions", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-audio-bytes", type=int, default=_MAX_AUDIO_BYTES)
    parser.add_argument("--language-mode", choices=("automatic", "explicit"), default="explicit")
    parser.add_argument("--language", default="en")
    parser.add_argument("--allow-status", action="append", type=int, dest="allowed_statuses")
    parser.add_argument(
        "--api-key-env",
        default=_DEFAULT_API_KEY_ENV,
        help="Environment variable containing the API key; the value is never printed.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Use only with an intentionally unauthenticated local API.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line client and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    api_key = None if args.no_auth else os.getenv(args.api_key_env)
    if not args.no_auth and api_key is None:
        parser.error(f"set {args.api_key_env} or pass --no-auth for a local unauthenticated API")
    language = None if args.language_mode == "automatic" else args.language
    try:
        config = LoadTestConfig(
            audio_path=args.audio,
            base_url=args.base_url,
            api_key=api_key,
            submissions=args.submissions,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
            max_audio_bytes=args.max_audio_bytes,
            language_mode=args.language_mode,
            language=language,
            allowed_statuses=frozenset(args.allowed_statuses or {202}),
        )
    except ValueError as exc:
        parser.error(str(exc))
    summary = asyncio.run(run_load(config))
    print(summary.to_json())
    return 0 if summary.passed else 1


if __name__ == "__main__":  # pragma: no cover - project script is the supported entry point
    raise SystemExit(main())
