"""Authenticated, bounded WebSocket transport for real-time dictation."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from speech_intelligence_api.application.live_transcription import (
    LiveTranscriptionService,
    LiveTranscriptionSession,
)
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.enums import (
    LanguageCode,
    LanguageSelectionMode,
    LiveEventType,
)
from speech_intelligence_api.domain.errors import (
    AuthenticationError,
    ErrorCode,
    RateLimitExceededError,
    ServiceError,
)
from speech_intelligence_api.domain.models import (
    LanguageSelection,
    LiveSessionEvent,
    LiveTranscriptionOptions,
)
from speech_intelligence_api.entrypoints.http.rate_limiting import (
    acquire_rate_limit,
    caller_identity,
)
from speech_intelligence_api.entrypoints.http.schemas import (
    LiveControlMessage,
    LiveErrorEventResponse,
    LivePongEventResponse,
    LiveReadyEventResponse,
    LiveSessionClosedEventResponse,
    LiveSpeechStartedEventResponse,
    LiveStartMessage,
    LiveTranscriptEventResponse,
    TranscriptSegmentResponse,
    TranscriptWordResponse,
)
from speech_intelligence_api.entrypoints.http.security import (
    ApiKeyAuthenticator,
    ApiPrincipal,
)
from speech_intelligence_api.ports.rate_limiting import RateLimiter

router = APIRouter(tags=["live-transcription"])

_CLOSE_AUTHENTICATION_FAILED = 4401
_CLOSE_ORIGIN_FORBIDDEN = 4403
_CLOSE_HANDSHAKE_TIMEOUT = 4408
_CLOSE_CAPACITY_EXCEEDED = 4429
_CLOSE_INVALID_MESSAGE = 1003
_CLOSE_MESSAGE_TOO_LARGE = 1009
_CLOSE_DEPENDENCY_FAILURE = 1011


@dataclass(frozen=True, slots=True)
class _Input:
    kind: Literal["audio", "commit", "stop", "ping", "wake"]
    audio: bytes | None = None


@dataclass(slots=True)
class _ReceiverState:
    disconnected: bool = False
    failure: _ProtocolFailure | None = None


class _ProtocolFailure(Exception):
    def __init__(self, code: str, detail: str, close_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.close_code = close_code


@router.websocket("/live-transcription", name="live_transcription")
async def live_transcription(websocket: WebSocket) -> None:
    """Accept raw 16 kHz mono PCM and emit partial/final native-script events."""

    settings: Settings = websocket.app.state.settings
    request_id = str(getattr(websocket.state, "request_id", uuid4().hex))
    if not settings.live_transcription_enabled:
        await websocket.close(code=_CLOSE_DEPENDENCY_FAILURE, reason="service unavailable")
        return
    if not _origin_allowed(websocket, settings):
        await websocket.close(code=_CLOSE_ORIGIN_FORBIDDEN, reason="origin forbidden")
        return

    await websocket.accept()
    try:
        start = await _receive_start(websocket, settings.live_handshake_timeout_seconds)
        principal = _authenticate(websocket, start)
        if settings.rate_limit_enabled:
            limiter: RateLimiter | None = websocket.app.state.rate_limiter
            await acquire_rate_limit(
                limiter,
                caller_identity(
                    principal,
                    websocket.client.host if websocket.client else None,
                ),
                scope="live",
                limit=settings.rate_limit_live_sessions,
                window_seconds=settings.rate_limit_window_seconds,
            )
        options = _options(start)
    except AuthenticationError as exc:
        await _send_error(
            websocket,
            request_id=request_id,
            code=exc.code.value,
            detail=exc.public_message,
            fatal=True,
        )
        await websocket.close(code=_CLOSE_AUTHENTICATION_FAILED, reason="authentication failed")
        return
    except RateLimitExceededError as exc:
        await _send_error(
            websocket,
            request_id=request_id,
            code=exc.code.value,
            detail=exc.public_message,
            details=exc.details,
            fatal=True,
        )
        await websocket.close(code=_CLOSE_CAPACITY_EXCEEDED, reason="rate limit exceeded")
        return
    except ServiceError as exc:
        await _send_error(
            websocket,
            request_id=request_id,
            code=exc.code.value,
            detail=exc.public_message,
            details=exc.details or None,
            fatal=True,
        )
        await websocket.close(code=_CLOSE_DEPENDENCY_FAILURE, reason="service unavailable")
        return
    except _ProtocolFailure as exc:
        await _send_protocol_failure(websocket, request_id, exc)
        return
    except ValueError:
        failure = _ProtocolFailure(
            ErrorCode.INVALID_REQUEST.value,
            "The live-transcription options are invalid.",
            _CLOSE_INVALID_MESSAGE,
        )
        await _send_protocol_failure(websocket, request_id, failure)
        return

    service: LiveTranscriptionService | None = websocket.app.state.live_transcription_service
    if service is None:
        await _send_error(
            websocket,
            request_id=request_id,
            code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
            detail="The live-transcription service is unavailable.",
            fatal=True,
        )
        await websocket.close(code=_CLOSE_DEPENDENCY_FAILURE, reason="service unavailable")
        return

    session_id = f"live_{uuid4().hex}"
    try:
        async with service.session(options) as session:
            await websocket.send_json(
                LiveReadyEventResponse(
                    request_id=request_id,
                    session_id=session_id,
                    expires_in_seconds=settings.live_max_session_seconds,
                    max_chunk_bytes=settings.live_max_chunk_bytes,
                ).model_dump(mode="json")
            )
            await _stream(websocket, session, settings, request_id, session_id)
    except ServiceError as exc:
        await _send_error(
            websocket,
            request_id=request_id,
            code=exc.code.value,
            detail=exc.public_message,
            details=exc.details or None,
            fatal=True,
        )
        close_code = (
            _CLOSE_CAPACITY_EXCEEDED
            if exc.code is ErrorCode.CAPACITY_EXCEEDED
            else _CLOSE_DEPENDENCY_FAILURE
        )
        with suppress(RuntimeError):
            await websocket.close(close_code, reason="live session unavailable")


async def _receive_start(websocket: WebSocket, timeout_seconds: float) -> LiveStartMessage:
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
    except TimeoutError:
        raise _ProtocolFailure(
            ErrorCode.INVALID_REQUEST.value,
            "The live-transcription handshake timed out.",
            _CLOSE_HANDSHAKE_TIMEOUT,
        ) from None
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if not isinstance(text, str):
        raise _ProtocolFailure(
            ErrorCode.INVALID_REQUEST.value,
            "The first WebSocket message must be a JSON start message.",
            _CLOSE_INVALID_MESSAGE,
        )
    try:
        return LiveStartMessage.model_validate_json(text)
    except ValidationError:
        raise _ProtocolFailure(
            ErrorCode.INVALID_REQUEST.value,
            "The live-transcription start message is invalid.",
            _CLOSE_INVALID_MESSAGE,
        ) from None


def _authenticate(websocket: WebSocket, start: LiveStartMessage) -> ApiPrincipal:
    authenticator: ApiKeyAuthenticator = websocket.app.state.authenticator
    message_key = start.api_key.get_secret_value() if start.api_key is not None else None
    return authenticator.authenticate(websocket.headers.get("X-API-Key") or message_key)


def _options(start: LiveStartMessage) -> LiveTranscriptionOptions:
    selected_language = (
        None if start.language_mode is LanguageSelectionMode.AUTOMATIC else start.language
    )
    selected_chinese_script = (
        start.chinese_script if selected_language in {None, LanguageCode.CHINESE} else None
    )
    return LiveTranscriptionOptions(
        language=LanguageSelection(
            mode=start.language_mode,
            language=selected_language,
            chinese_script=selected_chinese_script,
        ),
        sample_rate_hz=start.sample_rate_hz,
        vocabulary=tuple(start.vocabulary),
        word_timestamps=start.word_timestamps,
    )


async def _stream(
    websocket: WebSocket,
    session: LiveTranscriptionSession,
    settings: Settings,
    request_id: str,
    session_id: str,
) -> None:
    queue: asyncio.Queue[_Input] = asyncio.Queue(maxsize=settings.live_input_queue_chunks)
    state = _ReceiverState()
    receiver = asyncio.create_task(_receive_inputs(websocket, queue, state, settings))
    processor = asyncio.create_task(
        _process_inputs(websocket, session, queue, state, request_id, session_id)
    )
    done, _ = await asyncio.wait({receiver, processor}, return_when=asyncio.FIRST_COMPLETED)
    if processor in done:
        receiver.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await receiver
        await processor
        return
    await receiver
    await processor


async def _receive_inputs(
    websocket: WebSocket,
    queue: asyncio.Queue[_Input],
    state: _ReceiverState,
    settings: Settings,
) -> None:
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.live_idle_timeout_seconds,
                )
            except TimeoutError:
                state.failure = _ProtocolFailure(
                    ErrorCode.INVALID_REQUEST.value,
                    "The live-transcription session was idle for too long.",
                    _CLOSE_HANDSHAKE_TIMEOUT,
                )
                _wake(queue)
                return
            if message["type"] == "websocket.disconnect":
                state.disconnected = True
                _wake(queue)
                return
            audio = message.get("bytes")
            if isinstance(audio, bytes):
                if len(audio) > settings.live_max_chunk_bytes:
                    state.failure = _ProtocolFailure(
                        ErrorCode.PAYLOAD_TOO_LARGE.value,
                        "The live audio chunk exceeds the configured size limit.",
                        _CLOSE_MESSAGE_TOO_LARGE,
                    )
                    _wake(queue)
                    return
                if len(audio) % 2:
                    state.failure = _ProtocolFailure(
                        ErrorCode.INVALID_REQUEST.value,
                        "Live PCM chunks must contain whole signed 16-bit samples.",
                        _CLOSE_INVALID_MESSAGE,
                    )
                    _wake(queue)
                    return
                try:
                    queue.put_nowait(_Input("audio", audio))
                except asyncio.QueueFull:
                    state.failure = _ProtocolFailure(
                        ErrorCode.RATE_LIMITED.value,
                        "The client is streaming audio faster than it can be processed.",
                        _CLOSE_CAPACITY_EXCEEDED,
                    )
                    return
                continue
            text = message.get("text")
            if not isinstance(text, str):
                state.failure = _ProtocolFailure(
                    ErrorCode.INVALID_REQUEST.value,
                    "The WebSocket message type is not supported.",
                    _CLOSE_INVALID_MESSAGE,
                )
                _wake(queue)
                return
            try:
                control = LiveControlMessage.model_validate_json(text)
            except ValidationError:
                state.failure = _ProtocolFailure(
                    ErrorCode.INVALID_REQUEST.value,
                    "The live-transcription control message is invalid.",
                    _CLOSE_INVALID_MESSAGE,
                )
                _wake(queue)
                return
            try:
                queue.put_nowait(_Input(control.type))
            except asyncio.QueueFull:
                state.failure = _ProtocolFailure(
                    ErrorCode.RATE_LIMITED.value,
                    "The client is sending messages faster than they can be processed.",
                    _CLOSE_CAPACITY_EXCEEDED,
                )
                return
            if control.type == "stop":
                return
    except WebSocketDisconnect:
        state.disconnected = True
        _wake(queue)


async def _process_inputs(
    websocket: WebSocket,
    session: LiveTranscriptionSession,
    queue: asyncio.Queue[_Input],
    state: _ReceiverState,
    request_id: str,
    session_id: str,
) -> None:
    while True:
        if state.disconnected:
            await session.discard()
            return
        if state.failure is not None:
            await _send_protocol_failure(websocket, request_id, state.failure)
            return
        item = await queue.get()
        try:
            if item.kind == "audio":
                events = await session.ingest(item.audio or b"")
            elif item.kind == "commit":
                events = await session.commit()
            elif item.kind == "ping":
                await websocket.send_json(
                    LivePongEventResponse(
                        request_id=request_id,
                        session_id=session_id,
                    ).model_dump(mode="json")
                )
                continue
            elif item.kind == "stop":
                events = await session.close()
                await _send_events(websocket, events, request_id, session_id)
                await websocket.send_json(
                    LiveSessionClosedEventResponse(
                        request_id=request_id,
                        session_id=session_id,
                    ).model_dump(mode="json")
                )
                await websocket.close(code=1000, reason="session complete")
                return
            else:
                continue
            await _send_events(websocket, events, request_id, session_id)
        except ServiceError as exc:
            recoverable = exc.code in {ErrorCode.INVALID_AUDIO, ErrorCode.LANGUAGE_UNCERTAIN}
            await _send_error(
                websocket,
                request_id=request_id,
                code=exc.code.value,
                detail=exc.public_message,
                details=exc.details or None,
                fatal=not recoverable,
            )
            if not recoverable:
                await websocket.close(
                    code=_CLOSE_DEPENDENCY_FAILURE,
                    reason="live transcription failed",
                )
                return
        except ValueError:
            failure = _ProtocolFailure(
                ErrorCode.INVALID_REQUEST.value,
                "The live PCM stream is invalid.",
                _CLOSE_INVALID_MESSAGE,
            )
            await _send_protocol_failure(websocket, request_id, failure)
            return


async def _send_events(
    websocket: WebSocket,
    events: tuple[LiveSessionEvent, ...],
    request_id: str,
    session_id: str,
) -> None:
    for event in events:
        payload: LiveSpeechStartedEventResponse | LiveTranscriptEventResponse
        if event.event_type is LiveEventType.SPEECH_STARTED:
            payload = LiveSpeechStartedEventResponse(
                request_id=request_id,
                session_id=session_id,
                utterance_id=event.utterance_id,
                start_seconds=event.start_seconds,
            )
        else:
            result = event.result
            if result is None:
                raise RuntimeError("transcript event is missing its result")
            event_type: Literal["partial", "final"] = (
                "partial" if event.event_type is LiveEventType.PARTIAL else "final"
            )
            payload = LiveTranscriptEventResponse(
                type=event_type,
                request_id=request_id,
                session_id=session_id,
                utterance_id=event.utterance_id,
                revision=event.revision,
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                language=result.language,
                language_confidence_estimate=result.language_confidence_estimate,
                chinese_script=result.chinese_script,
                text=result.text,
                segments=[
                    TranscriptSegmentResponse(
                        text=segment.text,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        language=segment.language,
                        confidence_estimate=segment.confidence_estimate,
                        words=[
                            TranscriptWordResponse(
                                text=word.text,
                                start_seconds=word.start_seconds,
                                end_seconds=word.end_seconds,
                                confidence_estimate=word.confidence_estimate,
                            )
                            for word in segment.words
                        ],
                    )
                    for segment in result.segments
                ],
            )
        await websocket.send_json(payload.model_dump(mode="json"))


async def _send_protocol_failure(
    websocket: WebSocket,
    request_id: str,
    failure: _ProtocolFailure,
) -> None:
    await _send_error(
        websocket,
        request_id=request_id,
        code=failure.code,
        detail=failure.detail,
        fatal=True,
    )
    with suppress(RuntimeError):
        await websocket.close(code=failure.close_code, reason="invalid live session")


async def _send_error(
    websocket: WebSocket,
    *,
    request_id: str,
    code: str,
    detail: str,
    fatal: bool,
    details: dict[str, object] | None = None,
) -> None:
    await websocket.send_json(
        LiveErrorEventResponse(
            request_id=request_id,
            code=code,
            detail=detail,
            fatal=fatal,
            details=details,
        ).model_dump(mode="json")
    )


def _origin_allowed(websocket: WebSocket, settings: Settings) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    normalized = origin.rstrip("/")
    if normalized in settings.cors_allowed_origins:
        return True
    parsed = urlsplit(normalized)
    return parsed.netloc.casefold() == (websocket.headers.get("host") or "").casefold()


def _wake(queue: asyncio.Queue[_Input]) -> None:
    with suppress(asyncio.QueueFull):
        queue.put_nowait(_Input("wake"))
