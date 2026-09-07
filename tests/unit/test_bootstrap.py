"""Production composition tests that do not connect to external services."""

from pathlib import Path

import pytest

from speech_intelligence_api.bootstrap import (
    build_api_runtime,
    build_conversation_worker_runtime,
    build_worker_runtime,
)
from tests.factories import make_settings


def _resolved(path: Path) -> Path:
    return path.resolve()


@pytest.mark.asyncio
async def test_api_runtime_composes_optional_job_services(tmp_path: Path) -> None:
    settings = make_settings().model_copy(
        update={
            "async_jobs_enabled": True,
            "temp_storage_root": tmp_path,
        }
    )

    runtime = build_api_runtime(settings)

    assert runtime.transcription_service.jobs_enabled is True
    assert runtime.job_service is not None
    assert [check.name for check in runtime.readiness_checks] == ["redis", "redis_broker"]
    assert runtime.redis_client is not None
    assert runtime.broker_redis_client is not None
    assert runtime.cleanup_service is not None
    await runtime.close()


@pytest.mark.asyncio
async def test_api_runtime_composes_rate_limiting_without_async_jobs(tmp_path: Path) -> None:
    settings = make_settings().model_copy(
        update={
            "rate_limit_enabled": True,
            "temp_storage_root": tmp_path,
        }
    )

    runtime = build_api_runtime(settings)

    assert runtime.rate_limiter is not None
    assert runtime.redis_client is not None
    assert runtime.broker_redis_client is None
    assert [check.name for check in runtime.readiness_checks] == ["redis"]
    await runtime.close()


@pytest.mark.asyncio
async def test_worker_runtime_is_lazy_about_model_and_redis_io(tmp_path: Path) -> None:
    settings = make_settings().model_copy(
        update={
            "async_jobs_enabled": True,
            "temp_storage_root": tmp_path,
        }
    )

    runtime = build_worker_runtime(settings)

    assert runtime.service is not None
    assert runtime.blob_store.root == _resolved(tmp_path)
    assert runtime.cleanup_service is not None
    await runtime.redis_client.aclose()


@pytest.mark.asyncio
async def test_api_and_worker_compose_isolated_conversation_services(tmp_path: Path) -> None:
    settings = make_settings().model_copy(
        update={
            "async_jobs_enabled": True,
            "diarization_enabled": True,
            "temp_storage_root": tmp_path,
        }
    )

    api_runtime = build_api_runtime(settings)
    worker_runtime = build_conversation_worker_runtime(settings)

    assert api_runtime.conversation_service is not None
    assert worker_runtime.service is not None
    assert worker_runtime.blob_store.root == _resolved(tmp_path)
    await api_runtime.close()
    await worker_runtime.redis_client.aclose()


def test_conversation_worker_refuses_disabled_feature(tmp_path: Path) -> None:
    settings = make_settings().model_copy(update={"temp_storage_root": tmp_path})

    with pytest.raises(RuntimeError, match="diarization is disabled"):
        build_conversation_worker_runtime(settings)
