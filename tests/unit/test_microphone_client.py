"""Terminal microphone client tests without accessing host audio hardware."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest

import speech_intelligence_api.entrypoints.microphone_client as microphone_client
from speech_intelligence_api.domain.enums import LanguageCode
from speech_intelligence_api.entrypoints.microphone_client import (
    AudioBridge,
    MicrophoneClientConfig,
    MicrophoneClientError,
    _decode_event,
    _default_stream_factory,
    _device,
    _display_event,
    _run_session,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.incoming.put_nowait(
            json.dumps(
                {
                    "type": "ready",
                    "sample_rate_hz": 16_000,
                    "encoding": "pcm_s16le",
                }
            )
        )
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)
        if isinstance(message, bytes):
            self.incoming.put_nowait(json.dumps({"type": "partial", "text": "hello"}))
        elif message == '{"type":"stop"}':
            self.incoming.put_nowait(json.dumps({"type": "final", "text": "hello world"}))
            self.incoming.put_nowait(json.dumps({"type": "session_closed"}))

    async def recv(self) -> str | bytes:
        return await self.incoming.get()


class FakeStream:
    def __init__(
        self,
        callback: Callable[[bytes | bytearray | memoryview, int, object, object], None],
    ) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.callback(b"\x01\x00" * 160, 160, object(), "input overflow")

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    def __init__(self, *, working_device: int | None) -> None:
        self.working_device = working_device
        self.attempts: list[object] = []

    def query_devices(self) -> object:
        return [
            {"max_input_channels": 2, "default_samplerate": 48_000.0},
            {"max_input_channels": 4, "default_samplerate": 16_000.0},
            {"max_input_channels": 0, "default_samplerate": 16_000.0},
        ]

    def RawInputStream(self, **kwargs: object) -> FakeStream:
        device = kwargs["device"]
        self.attempts.append(device)
        if device != self.working_device:
            raise RuntimeError("driver rejected format")
        callback = kwargs["callback"]
        assert callable(callback)
        return FakeStream(callback)


def _config(**overrides: object) -> MicrophoneClientConfig:
    values: dict[str, object] = {
        "url": "ws://127.0.0.1:8000/v1/live-transcription",
        "duration_seconds": 0.02,
        "chunk_ms": 100,
        "language": LanguageCode.ENGLISH,
    }
    values.update(overrides)
    return MicrophoneClientConfig(**values)  # type: ignore[arg-type]


def test_microphone_config_builds_explicit_secret_bearing_start_message() -> None:
    config = _config(api_key="private-key", vocabulary=("FastAPI",))

    assert config.block_size == 1600
    assert config.start_payload() == {
        "type": "start",
        "encoding": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "language_mode": "explicit",
        "language": "en",
        "vocabulary": ["FastAPI"],
        "word_timestamps": True,
        "api_key": "private-key",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"url": "http://localhost/live"}, "ws://"),
        ({"duration_seconds": 0}, "duration"),
        ({"final_timeout_seconds": 0}, "final-timeout"),
        ({"chunk_ms": 10}, "chunk-ms"),
        ({"vocabulary": ("",)}, "vocabulary"),
    ],
)
def test_microphone_config_rejects_invalid_options(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_event_decoder_and_renderer_reject_invalid_server_data() -> None:
    output: list[str] = []

    assert _display_event({"type": "speech_started"}, output.append) is False
    assert _display_event({"type": "partial", "text": "hel"}, output.append) is False
    assert _display_event({"type": "final", "text": "hello"}, output.append) is False
    assert _display_event({"type": "session_closed"}, output.append) is True
    assert output == [
        "[listening] Speech detected",
        "[partial] hel",
        "[final]   hello",
    ]
    with pytest.raises(MicrophoneClientError, match="binary"):
        _decode_event(b"binary")
    with pytest.raises(MicrophoneClientError, match="invalid JSON"):
        _decode_event("not-json")
    with pytest.raises(MicrophoneClientError, match="invalid event"):
        _decode_event("[]")


def test_fatal_error_event_is_sanitized_and_raised() -> None:
    output: list[str] = []

    with pytest.raises(MicrophoneClientError, match="Unavailable"):
        _display_event(
            {"type": "error", "detail": "Unavailable", "fatal": True},
            output.append,
        )

    assert output == ["[error]   Unavailable"]


@pytest.mark.asyncio
async def test_audio_bridge_detects_overflow_and_finishes_idempotently() -> None:
    bridge = AudioBridge(asyncio.get_running_loop(), max_chunks=1)
    bridge._enqueue(b"first", None)
    bridge._enqueue(b"second", None)

    with pytest.raises(MicrophoneClientError, match="overflowed"):
        await bridge.receive()
    await bridge.finish(discard_pending=True)
    await bridge.finish(discard_pending=True)

    bridge = AudioBridge(asyncio.get_running_loop(), max_chunks=1)
    await bridge.finish()
    await bridge.finish()
    assert await bridge.receive() is None


@pytest.mark.asyncio
async def test_live_microphone_session_streams_pcm_and_prints_results() -> None:
    websocket = FakeWebSocket()
    output: list[str] = []
    streams: list[FakeStream] = []

    def stream_factory(bridge: AudioBridge, config: MicrophoneClientConfig) -> FakeStream:
        assert config.block_size == 1600
        stream = FakeStream(bridge.callback)
        streams.append(stream)
        return stream

    await _run_session(websocket, _config(), output=output.append, stream_factory=stream_factory)

    assert streams[0].started is True
    assert streams[0].stopped is True
    assert streams[0].closed is True
    assert any(isinstance(message, bytes) for message in websocket.sent)
    assert websocket.sent[-1] == '{"type":"stop"}'
    start = json.loads(str(websocket.sent[0]))
    assert start["language_mode"] == "explicit"
    assert "api_key" not in start
    assert output == [
        "Connected: 16000 Hz pcm_s16le. Speak for 0.02 seconds...",
        "[warning] Microphone status: input overflow",
        "[partial] hello",
        "Capture complete. Waiting for final transcription...",
        "[final]   hello world",
    ]


def test_default_stream_prefers_native_16khz_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sounddevice = FakeSoundDevice(working_device=1)
    monkeypatch.setattr(microphone_client, "_load_sounddevice", lambda: sounddevice)
    loop = asyncio.new_event_loop()
    bridge = AudioBridge(loop, max_chunks=2)

    try:
        stream = _default_stream_factory(bridge, _config(language=None))
    finally:
        loop.close()

    assert isinstance(stream, FakeStream)
    assert sounddevice.attempts == [1]


def test_explicit_incompatible_device_returns_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sounddevice = FakeSoundDevice(working_device=None)
    monkeypatch.setattr(microphone_client, "_load_sounddevice", lambda: sounddevice)
    loop = asyncio.new_event_loop()
    bridge = AudioBridge(loop, max_chunks=2)

    try:
        with pytest.raises(MicrophoneClientError, match="--list-devices"):
            _default_stream_factory(bridge, _config(device=9))
    finally:
        loop.close()

    assert sounddevice.attempts == [9]


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("2", 2), ("USB Mic", "USB Mic")])
def test_device_argument_supports_index_or_name(raw: str | None, expected: object) -> None:
    assert _device(raw) == expected
