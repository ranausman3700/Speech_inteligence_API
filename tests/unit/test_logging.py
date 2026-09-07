"""Structured logging privacy tests."""

import json
import logging

from speech_intelligence_api.logging import JsonFormatter, redact_sensitive_data


def test_recursive_redaction_preserves_safe_metadata() -> None:
    redacted = redact_sensitive_data(
        {
            "job_id": "job_1",
            "api_key": "raw-key",
            "nested": {"transcript": "private words", "language": "ar"},
            "items": [{"authorization": "Bearer secret"}],
        }
    )

    assert redacted == {
        "job_id": "job_1",
        "api_key": "[REDACTED]",
        "nested": {"transcript": "[REDACTED]", "language": "ar"},
        "items": [{"authorization": "[REDACTED]"}],
    }


def test_json_formatter_redacts_extra_fields() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Processed job",
        args=(),
        exc_info=None,
    )
    record.job_id = "job_1"
    record.audio_path = "private.wav"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "Processed job"
    assert payload["context"]["job_id"] == "job_1"
    assert payload["context"]["audio_path"] == "[REDACTED]"
