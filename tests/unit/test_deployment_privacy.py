"""Deployment contracts that keep private Redis state off persistent storage."""

from pathlib import Path


def _project_file(name: str) -> str:
    return (Path(__file__).parents[2] / name).read_text(encoding="utf-8")


def test_compose_separates_ephemeral_state_from_durable_opaque_broker() -> None:
    compose = _project_file("compose.yaml")
    state = compose.split("\n  redis-state:\n", 1)[1].split("\n  redis-broker:\n", 1)[0]
    broker = compose.split("\n  redis-broker:\n", 1)[1].split("\n  api:\n", 1)[0]

    assert "redis://redis-state:6379/0" in compose
    assert "redis://redis-broker:6379/0" in compose
    assert '      - "no"' in state
    assert '      - ""' in state
    assert "/data:size=1024m" in state
    assert "volumes:" not in state
    assert '      - "yes"' in broker
    assert "redis-broker-data:/data" in broker
    assert "redis-data" not in compose


def test_environment_example_uses_split_roles_and_cleanup_window() -> None:
    example = _project_file(".env.example")

    assert "SPEECH_API_REDIS_JOB_URL=redis://redis-state:6379/0" in example
    assert "SPEECH_API_CELERY_BROKER_URL=redis://redis-broker:6379/0" in example
    assert "SPEECH_API_ARTIFACT_CLEANUP_INTERVAL_SECONDS=30" in example
