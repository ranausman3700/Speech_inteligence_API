"""HTTP correlation and privacy-safe access logging."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import cast
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from speech_intelligence_api.domain.errors import PayloadTooLargeError
from speech_intelligence_api.entrypoints.http.errors import service_error_response
from speech_intelligence_api.logging import request_id_context
from speech_intelligence_api.ports.observability import Observability

logger = logging.getLogger(__name__)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_TRACE_HEADERS = frozenset({b"traceparent"})


class ObservabilityMiddleware:
    """Measure route templates and continue W3C traces without reading payloads."""

    def __init__(self, app: ASGIApp, *, observability: Observability) -> None:
        self.app = app
        self.observability = observability

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        is_http = scope["type"] == "http"
        method = str(scope.get("method", "WEBSOCKET")).upper()
        status_code = 500 if is_http else 101
        close_code = 1006
        started_at = time.perf_counter()
        carrier = self._trace_carrier(scope)
        attributes = (
            {"http.request.method": method} if is_http else {"network.protocol.name": "websocket"}
        )

        async def observed_send(message: Message) -> None:
            nonlocal close_code, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "websocket.close":
                close_code = message.get("code", 1000)
            await send(message)

        with self.observability.span(
            f"{method} request",
            kind="server",
            attributes=attributes,
            incoming_carrier=carrier,
        ) as span:
            try:
                await self.app(scope, receive, observed_send)
            finally:
                route = _route_template(scope)
                duration_seconds = time.perf_counter() - started_at
                span.update_name(f"{method} {route}")
                if is_http:
                    span.set_attribute("http.route", route)
                    span.set_attribute("http.response.status_code", status_code)
                    if status_code >= 500:
                        span.mark_error()
                    self.observability.record_http_request(
                        method=method,
                        route=route,
                        status_code=status_code,
                        duration_seconds=duration_seconds,
                    )
                else:
                    span.set_attribute("network.protocol.name", "websocket")
                    span.set_attribute("websocket.close.code", close_code)
                    if close_code >= 4000:
                        span.mark_error()

    @staticmethod
    def _trace_carrier(scope: Scope) -> dict[str, str]:
        return {
            name.decode("latin-1"): value.decode("latin-1")
            for name, value in scope.get("headers", [])
            if name.lower() in _TRACE_HEADERS
        }


class SecurityHeadersMiddleware:
    """Prevent browser caching/sniffing and advertise HTTPS transport security."""

    def __init__(self, app: ASGIApp, *, hsts_max_age_seconds: int) -> None:
        self.app = app
        self.hsts_max_age_seconds = hsts_max_age_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if "cache-control" not in headers:
                    headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=()"
                if scope.get("scheme") == "https" and self.hsts_max_age_seconds > 0:
                    headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age_seconds}"
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class RequestBodyLimitMiddleware:
    """Bound multipart request bodies before Starlette can spool unlimited input."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        path: str,
        max_body_bytes: int,
        max_upload_bytes: int,
    ) -> None:
        self.app = app
        self.path = path
        self.max_body_bytes = max_body_bytes
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.path
        ):
            await self.app(scope, receive, send)
            return

        declared_size = self._declared_content_length(scope)
        if declared_size is not None and declared_size > self.max_body_bytes:
            await self._send_payload_too_large(scope, receive, send)
            return

        received_size = 0
        overflowed = False

        async def limited_receive() -> Message:
            nonlocal overflowed, received_size
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > self.max_body_bytes:
                    overflowed = True
                    raise PayloadTooLargeError(self.max_upload_bytes)
            return message

        buffered_messages: list[Message] = []

        async def buffered_send(message: Message) -> None:
            buffered_messages.append(message)

        try:
            await self.app(scope, limited_receive, buffered_send)
        except PayloadTooLargeError:
            overflowed = True

        if overflowed:
            await self._send_payload_too_large(scope, receive, send)
            return
        for message in buffered_messages:
            await send(message)

    @staticmethod
    def _declared_content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def _send_payload_too_large(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        request = Request(scope, receive=receive)
        response = service_error_response(request, PayloadTooLargeError(self.max_upload_bytes))
        await response(scope, receive, send)


class RequestIdMiddleware:
    """Propagate a safe request ID and emit metadata-only access logs."""

    def __init__(self, app: ASGIApp, *, header_name: str = "X-Request-ID") -> None:
        self.app = app
        self.header_name = header_name
        self._header_name_bytes = header_name.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        supplied_id = self._read_request_id(scope)
        request_id = (
            supplied_id if supplied_id and _SAFE_REQUEST_ID.fullmatch(supplied_id) else uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_context.set(request_id)
        started_at = time.perf_counter()
        status_code = 101 if scope["type"] == "websocket" else 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[self.header_name] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "Request completed",
                extra={
                    "method": scope.get("method", "WEBSOCKET"),
                    "route": self._route_template(scope),
                    "status_code": status_code,
                    "duration_ms": elapsed_ms,
                },
            )
            request_id_context.reset(token)

    def _read_request_id(self, scope: Scope) -> str | None:
        for name, value in scope.get("headers", []):
            if name.lower() == self._header_name_bytes:
                return cast(bytes, value).decode("latin-1")
        return None

    @staticmethod
    def _route_template(scope: Scope) -> str:
        return _route_template(scope)


class ControlCenterUsageMiddleware:
    """Reports request outcomes to the CodesBunny API Control Center dashboard.

    Inactive unless `SPEECH_API_CONTROL_CENTER_ENABLED=true` (see `app.state.
    control_center_client`). The report is sent as a fire-and-forget
    background task so a slow or unreachable Control Center never adds
    latency to this API's own responses.
    """

    _API_KEY_HEADER = b"x-api-key"
    _USER_AGENT_HEADER = b"user-agent"
    _FORWARDED_FOR_HEADER = b"x-forwarded-for"

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        control_center = scope["app"].state.control_center_client
        if control_center is None:
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        status_code = 500

        async def observed_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        finally:
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            headers = scope.get("headers", [])
            token = self._header_value(headers, self._API_KEY_HEADER)
            user_agent = self._header_value(headers, self._USER_AGENT_HEADER)
            forwarded_for = self._header_value(headers, self._FORWARDED_FOR_HEADER)
            client = scope.get("client")
            ip_address = (
                forwarded_for.split(",")[0].strip()
                if forwarded_for
                else (client[0] if client else None)
            )

            app_state = scope["app"].state
            pending_tasks: set[asyncio.Task[None]] = app_state.control_center_tasks
            task = asyncio.create_task(
                control_center.log_usage(
                    token=token,
                    endpoint=_route_template(scope),
                    method=str(scope.get("method", "GET")),
                    status_code=status_code,
                    response_time_ms=duration_ms,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            )
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

    @staticmethod
    def _header_value(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
        for header_name, value in headers:
            if header_name.lower() == name:
                return value.decode("latin-1")
        return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else "unmatched"
