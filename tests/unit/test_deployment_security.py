"""Production deployment security contracts."""

from pathlib import Path


def _project_file(name: str) -> str:
    return (Path(__file__).parents[2] / name).read_text(encoding="utf-8")


def test_compose_binds_api_to_loopback_and_trusts_only_configured_proxies() -> None:
    compose = _project_file("compose.yaml")

    assert "${SPEECH_API_BIND_ADDRESS:-127.0.0.1}" in compose
    assert "${SPEECH_API_HTTP_PORT:-8000}:8000" in compose
    assert "${SPEECH_API_FORWARDED_ALLOW_IPS:-127.0.0.1}" in compose
    assert '"8000:8000"' not in compose


def test_compose_drops_capabilities_and_bounds_processes() -> None:
    compose = _project_file("compose.yaml")

    assert compose.count("cap_drop:") == 7
    assert compose.count("    - ALL") == 7
    assert compose.count("pids_limit:") == 7
    assert compose.count("init: true") == 7
    assert "no-new-privileges:true" in compose


def test_redis_roles_are_not_published_to_the_host() -> None:
    compose = _project_file("compose.yaml")
    state = compose.split("\n  redis-state:\n", 1)[1].split("\n  redis-broker:\n", 1)[0]
    broker = compose.split("\n  redis-broker:\n", 1)[1].split("\n  api:\n", 1)[0]

    assert "ports:" not in state
    assert "ports:" not in broker
    assert 'user: "999:1000"' in state
    assert 'user: "999:1000"' in broker
    assert "/data:size=1024m,mode=0700,uid=999,gid=1000" in state


def test_all_runtime_images_drop_root_before_starting() -> None:
    for name in (
        "Dockerfile",
        "Dockerfile.gpu",
        "Dockerfile.diarization",
        "Dockerfile.diarization.cpu",
    ):
        dockerfile = _project_file(name)

        assert "USER app" in dockerfile


def test_cpu_diarization_profile_is_isolated_and_gpu_free() -> None:
    compose = _project_file("compose.yaml")
    service = compose.split("\n  worker-diarization-cpu:\n", 1)[1].split("\n  beat:\n", 1)[0]
    dockerfile = _project_file("Dockerfile.diarization.cpu")

    assert 'profiles: ["diarization-cpu"]' in service
    assert "dockerfile: Dockerfile.diarization.cpu" in service
    assert 'SPEECH_API_DIARIZATION_ENABLED: "true"' in service
    assert "SPEECH_API_ASR_DEVICE: cpu" in service
    assert "SPEECH_API_ASR_COMPUTE_TYPE: int8" in service
    assert "SPEECH_API_DIARIZATION_DEVICE: cpu" in service
    assert "MPLCONFIGDIR: /tmp/matplotlib" in service
    assert "capabilities: [gpu]" not in service
    assert "nvidia/cuda" not in dockerfile
    assert "optional-dependencies']['diarization']" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.11.0+cpu" in dockerfile
    assert "torchaudio==2.11.0+cpu" in dockerfile
    assert "torchcodec==0.11.1+cpu" in dockerfile
    assert "ffmpeg libgomp1 libsndfile1" in dockerfile


def test_production_example_documents_http_security_boundary() -> None:
    example = _project_file(".env.example")

    assert "SPEECH_API_TRUSTED_HOSTS=" in example
    assert "SPEECH_API_SECURITY_HEADERS_ENABLED=true" in example
    assert "SPEECH_API_HSTS_MAX_AGE_SECONDS=31536000" in example
    assert "SPEECH_API_BIND_ADDRESS=127.0.0.1" in example
    assert "SPEECH_API_FORWARDED_ALLOW_IPS=127.0.0.1" in example


def test_container_context_excludes_secrets_and_repository_metadata() -> None:
    dockerignore = _project_file(".dockerignore").splitlines()

    assert ".env" in dockerignore
    assert ".git" in dockerignore


def test_ci_runs_complete_release_quality_gates() -> None:
    workflow = _project_file(".github/workflows/ci.yml")

    assert "python -m pip check" in workflow
    assert "python -m ruff check ." in workflow
    assert "python -m ruff format --check ." in workflow
    assert "python -m mypy src tests" in workflow
    assert "python -m pytest" in workflow
    assert "docker compose config --quiet" in workflow
    assert "python -m pip wheel --no-deps --wheel-dir dist ." in workflow
    assert "speech-intelligence-api:phase-8" in workflow
