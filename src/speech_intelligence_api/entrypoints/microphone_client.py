"""Terminal microphone client for manually exercising live transcription."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from speech_intelligence_api.domain.enums import LanguageCode

_SAMPLE_RATE_HZ = 16_000
_BYTES_PER_SAMPLE = 2


class MicrophoneClientError(RuntimeError):
    """Safe terminal-facing client failure."""


class AudioInputStream(Protocol):
    """Small sounddevice stream boundary used by the client and tests."""

    def start(self) -> object: ...

    def stop(self) -> object: ...

    def close(self) -> object: ...


class SoundDeviceModule(Protocol):
    """Subset of sounddevice loaded only for the optional microphone command."""

    def RawInputStream(self, **kwargs: object) -> AudioInputStream: ...

    def query_devices(self) -> object: ...


class WebSocketClient(Protocol):
    """Transport subset needed by the live microphone session."""

    async def send(self, message: str | bytes) -> None: ...

    async def recv(self) -> str | bytes: ...


Output = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MicrophoneClientConfig:
    """Validated terminal-client options."""

    url: str
    duration_seconds: float
    chunk_ms: int
    final_timeout_seconds: float = 600
    api_key: str | None = None
    language: LanguageCode | None = None
    vocabulary: tuple[str, ...] = ()
    word_timestamps: bool = True
    device: str | int | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("--url must be an absolute ws:// or wss:// URL")
        if self.duration_seconds <= 0:
            raise ValueError("--duration must be greater than zero")
        if self.final_timeout_seconds <= 0:
            raise ValueError("--final-timeout must be greater than zero")
        if not 20 <= self.chunk_ms <= 500:
            raise ValueError("--chunk-ms must be between 20 and 500")
        if len(self.vocabulary) > 100 or any(not item.strip() for item in self.vocabulary):
            raise ValueError("--vocabulary accepts at most 100 non-empty terms")

    @property
    def block_size(self) -> int:
        """Samples per microphone callback."""

        return int(_SAMPLE_RATE_HZ * self.chunk_ms / 1000)

    def start_payload(self) -> dict[str, object]:
        """Build the first authenticated WebSocket message without logging secrets."""

        payload: dict[str, object] = {
            "type": "start",
            "encoding": "pcm_s16le",
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "language_mode": "explicit" if self.language is not None else "automatic",
            "language": self.language.value if self.language is not None else None,
            "vocabulary": list(self.vocabulary),
            "word_timestamps": self.word_timestamps,
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        return payload


class AudioBridge:
    """Bounded thread-to-async bridge for non-blocking microphone callbacks."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, max_chunks: int) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max_chunks)
        self._overflow = False
        self._stopped = False
        self._status: str | None = None

    def callback(
        self,
        indata: bytes | bytearray | memoryview,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Copy one raw frame and schedule it on the owning event loop."""

        del frames, time_info
        chunk = bytes(indata)
        self._loop.call_soon_threadsafe(self._enqueue, chunk, str(status) if status else None)

    def _enqueue(self, chunk: bytes, status: str | None) -> None:
        if self._stopped:
            return
        if status:
            self._status = status
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self._overflow = True

    async def receive(self) -> bytes | None:
        """Return the next frame and fail rather than silently dropping speech."""

        if self._overflow:
            raise MicrophoneClientError("Microphone audio overflowed before it could be sent")
        chunk = await self._queue.get()
        if self._overflow:
            raise MicrophoneClientError("Microphone audio overflowed before it could be sent")
        return chunk

    def pop_status(self) -> str | None:
        status, self._status = self._status, None
        return status

    async def finish(self, *, discard_pending: bool = False) -> None:
        """Stop accepting callback data and wake the sender."""

        if self._stopped:
            return
        self._stopped = True
        if discard_pending:
            while not self._queue.empty():
                self._queue.get_nowait()
        await self._queue.put(None)


StreamFactory = Callable[[AudioBridge, MicrophoneClientConfig], AudioInputStream]


def _load_sounddevice() -> SoundDeviceModule:
    try:
        module = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise MicrophoneClientError(
            'Microphone support is not installed. Run: python -m pip install -e ".[microphone]"'
        ) from exc
    return cast(SoundDeviceModule, module)


def _default_stream_factory(
    bridge: AudioBridge,
    config: MicrophoneClientConfig,
) -> AudioInputStream:
    sounddevice = _load_sounddevice()

    def open_device(device: str | int | None) -> AudioInputStream:
        return sounddevice.RawInputStream(
            samplerate=_SAMPLE_RATE_HZ,
            blocksize=config.block_size,
            device=device,
            channels=1,
            dtype="int16",
            callback=bridge.callback,
        )

    if config.device is not None:
        try:
            return open_device(config.device)
        except Exception as exc:
            raise MicrophoneClientError(
                f"Unable to open input device {config.device!r} at 16 kHz. "
                "Run --list-devices and select a compatible input device."
            ) from exc

    devices = sounddevice.query_devices()
    if not isinstance(devices, Sequence):
        raise MicrophoneClientError("Unable to inspect host microphone devices")
    candidates: list[tuple[int, bool]] = []
    for index, device_info in enumerate(devices):
        if not isinstance(device_info, Mapping):
            continue
        max_inputs = device_info.get("max_input_channels")
        default_rate = device_info.get("default_samplerate")
        if isinstance(max_inputs, int | float) and max_inputs >= 1:
            candidates.append((index, default_rate == _SAMPLE_RATE_HZ))
    candidates.sort(key=lambda candidate: not candidate[1])
    last_error: Exception | None = None
    for index, _ in candidates:
        try:
            return open_device(index)
        except Exception as exc:
            last_error = exc
    raise MicrophoneClientError(
        "No 16 kHz input device could be opened. Run --list-devices and pass --device INDEX."
    ) from last_error


def _close_stream(stream: AudioInputStream) -> None:
    try:
        stream.stop()
    finally:
        stream.close()


def _decode_event(message: str | bytes) -> dict[str, object]:
    if not isinstance(message, str):
        raise MicrophoneClientError("Server returned an unexpected binary message")
    try:
        event = json.loads(message)
    except json.JSONDecodeError as exc:
        raise MicrophoneClientError("Server returned invalid JSON") from exc
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise MicrophoneClientError("Server returned an invalid event")
    return cast(dict[str, object], event)


def _display_event(event: dict[str, object], output: Output) -> bool:
    event_type = event["type"]
    if event_type == "speech_started":
        output("[listening] Speech detected")
    elif event_type == "partial":
        output(f"[partial] {event.get('text', '')}")
    elif event_type == "final":
        output(f"[final]   {event.get('text', '')}")
    elif event_type == "error":
        detail = str(event.get("detail", "Live transcription failed"))
        output(f"[error]   {detail}")
        if event.get("fatal") is True:
            raise MicrophoneClientError(detail)
    return event_type == "session_closed"


async def _send_audio(websocket: WebSocketClient, bridge: AudioBridge, output: Output) -> None:
    while True:
        chunk = await bridge.receive()
        if chunk is None:
            return
        status = bridge.pop_status()
        if status:
            output(f"[warning] Microphone status: {status}")
        await websocket.send(chunk)


async def _receive_events(websocket: WebSocketClient, output: Output) -> None:
    try:
        while True:
            event = _decode_event(await websocket.recv())
            if _display_event(event, output):
                return
    except ConnectionClosed as exc:
        raise MicrophoneClientError("WebSocket closed before the session completed") from exc


async def _run_session(
    websocket: WebSocketClient,
    config: MicrophoneClientConfig,
    *,
    output: Output,
    stream_factory: StreamFactory = _default_stream_factory,
) -> None:
    await websocket.send(json.dumps(config.start_payload(), separators=(",", ":")))
    try:
        ready = _decode_event(await asyncio.wait_for(websocket.recv(), timeout=10))
    except TimeoutError as exc:
        raise MicrophoneClientError(
            "Server did not accept the live session within 10 seconds"
        ) from exc
    if ready["type"] == "error":
        _display_event(ready, output)
        raise MicrophoneClientError(str(ready.get("detail", "Live session was rejected")))
    if ready["type"] != "ready":
        raise MicrophoneClientError("Server did not return a ready event")

    output(
        f"Connected: {ready.get('sample_rate_hz')} Hz {ready.get('encoding')}. "
        f"Speak for {config.duration_seconds:g} seconds..."
    )
    loop = asyncio.get_running_loop()
    bridge = AudioBridge(loop, max_chunks=max(10, 5_000 // config.chunk_ms))
    stream = stream_factory(bridge, config)
    await asyncio.to_thread(stream.start)
    sender = asyncio.create_task(_send_audio(websocket, bridge, output))
    receiver = asyncio.create_task(_receive_events(websocket, output))
    timer = asyncio.create_task(asyncio.sleep(config.duration_seconds))
    server_finished = False
    try:
        done, _ = await asyncio.wait(
            {sender, receiver, timer},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sender in done:
            await sender
        if receiver in done:
            server_finished = True
            await receiver
    finally:
        timer.cancel()
        with suppress(asyncio.CancelledError):
            await timer
        await asyncio.to_thread(_close_stream, stream)
        await bridge.finish(discard_pending=sender.done())
        try:
            await sender
        except BaseException:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver
            raise

    if server_finished:
        return
    output("Capture complete. Waiting for final transcription...")
    await websocket.send('{"type":"stop"}')
    try:
        await asyncio.wait_for(receiver, timeout=config.final_timeout_seconds)
    except TimeoutError as exc:
        receiver.cancel()
        with suppress(asyncio.CancelledError):
            await receiver
        raise MicrophoneClientError(
            "Server did not return final transcription before --final-timeout"
        ) from exc


async def run_client(config: MicrophoneClientConfig, *, output: Output = print) -> None:
    """Connect to the live endpoint and stream one bounded microphone session."""

    try:
        async with connect(
            config.url,
            open_timeout=10,
            close_timeout=5,
            max_size=1_048_576,
        ) as websocket:
            await _run_session(websocket, config, output=output)
    except (OSError, ConnectionClosed) as exc:
        raise MicrophoneClientError(f"Unable to use the live WebSocket: {exc}") from exc


def _device(value: str | None) -> str | int | None:
    if value is None:
        return None
    return int(value) if value.isdecimal() else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream microphone speech to live transcription")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/live-transcription")
    parser.add_argument("--api-key", default=os.getenv("SPEECH_API_CLIENT_API_KEY"))
    parser.add_argument("--language", choices=[item.value for item in LanguageCode])
    parser.add_argument("--duration", type=float, default=30.0, dest="duration_seconds")
    parser.add_argument("--final-timeout", type=float, default=600.0, dest="final_timeout_seconds")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--device", help="Input-device index or name")
    parser.add_argument("--vocabulary", action="append", default=[])
    parser.add_argument("--no-word-timestamps", action="store_false", dest="word_timestamps")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point with sanitized failures and no secret logging."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.list_devices:
            print(_load_sounddevice().query_devices())
            return 0
        config = MicrophoneClientConfig(
            url=args.url,
            duration_seconds=args.duration_seconds,
            chunk_ms=args.chunk_ms,
            final_timeout_seconds=args.final_timeout_seconds,
            api_key=args.api_key,
            language=LanguageCode(args.language) if args.language else None,
            vocabulary=tuple(args.vocabulary),
            word_timestamps=args.word_timestamps,
            device=_device(args.device),
        )
        asyncio.run(run_client(config))
    except (MicrophoneClientError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("\nStopped by user")
    return 0
