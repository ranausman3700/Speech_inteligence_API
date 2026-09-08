"""FastAPI application factory."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, status
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from speech_intelligence_api import __version__
from speech_intelligence_api.adapters.observability import build_observability
from speech_intelligence_api.adapters.text_documents import PlainTextDocumentRenderer
from speech_intelligence_api.application.conversations import ConversationSubmissionService
from speech_intelligence_api.application.exports import TranscriptExportService
from speech_intelligence_api.application.jobs import JobManagementService
from speech_intelligence_api.application.live_transcription import LiveTranscriptionService
from speech_intelligence_api.application.readiness import ReadinessCheck, ReadinessService
from speech_intelligence_api.application.transcriptions import BatchTranscriptionService
from speech_intelligence_api.bootstrap import ApiRuntime, build_api_runtime
from speech_intelligence_api.config import Settings, get_settings
from speech_intelligence_api.domain.enums import ExportFormat
from speech_intelligence_api.entrypoints.http.dependencies import authenticate_api_key
from speech_intelligence_api.entrypoints.http.errors import install_exception_handlers
from speech_intelligence_api.entrypoints.http.middleware import (
    ObservabilityMiddleware,
    RequestBodyLimitMiddleware,
    RequestIdMiddleware,
    SecurityHeadersMiddleware,
)
from speech_intelligence_api.entrypoints.http.routes import health, live, metrics, v1
from speech_intelligence_api.entrypoints.http.schemas import ProblemDetail
from speech_intelligence_api.entrypoints.http.security import ApiKeyAuthenticator
from speech_intelligence_api.logging import configure_logging
from speech_intelligence_api.ports.documents import DocumentRenderer
from speech_intelligence_api.ports.observability import Observability
from speech_intelligence_api.ports.rate_limiting import RateLimiter

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Sequence[ReadinessCheck] = (),
    transcription_service: BatchTranscriptionService | None = None,
    conversation_service: ConversationSubmissionService | None = None,
    live_transcription_service: LiveTranscriptionService | None = None,
    job_service: JobManagementService | None = None,
    rate_limiter: RateLimiter | None = None,
    observability: Observability | None = None,
) -> FastAPI:
    """Build an independently testable, side-effect-minimized FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    owns_observability = observability is None
    resolved_observability = observability or build_observability(resolved_settings)
    runtime: ApiRuntime | None = None
    if transcription_service is None:
        runtime = build_api_runtime(
            resolved_settings,
            observability=resolved_observability,
        )
        transcription_service = runtime.transcription_service
        conversation_service = runtime.conversation_service
        live_transcription_service = runtime.live_transcription_service
        job_service = runtime.job_service
        rate_limiter = runtime.rate_limiter
        readiness_checks = (*readiness_checks, *runtime.readiness_checks)

    docs_url = "/docs" if resolved_settings.docs_enabled else None
    openapi_url = "/openapi.json" if resolved_settings.docs_enabled else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None and runtime.cleanup_service is not None:
            await runtime.cleanup_service.execute()
        app.state.started = True
        try:
            yield
        finally:
            app.state.started = False
            try:
                if runtime is not None:
                    await runtime.close()
            finally:
                if owns_observability:
                    await resolved_observability.close()

    app = FastAPI(
        title="Speech Intelligence API",
        summary="Native-script speech intelligence backend",
        description=(
            "Backend-only API for transcription, real-time dictation, and speaker diarization."
        ),
        version=resolved_settings.service_version,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
        contact={"name": "Speech Intelligence API operators"},
        license_info={"name": "MIT"},
    )
    app.state.settings = resolved_settings
    app.state.authenticator = ApiKeyAuthenticator(resolved_settings)
    app.state.readiness_service = ReadinessService(tuple(readiness_checks))
    app.state.transcription_service = transcription_service
    app.state.export_service = _export_service(resolved_settings)
    app.state.conversation_service = conversation_service
    app.state.live_transcription_service = live_transcription_service
    app.state.job_service = job_service
    app.state.rate_limiter = rate_limiter
    app.state.observability = resolved_observability

    if resolved_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Idempotency-Key",
                "traceparent",
                "X-API-Key",
                resolved_settings.request_id_header,
            ],
            expose_headers=[
                "Retry-After",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset-After",
                resolved_settings.request_id_header,
            ],
        )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(resolved_settings.trusted_hosts),
        www_redirect=False,
    )

    app.add_middleware(
        RequestBodyLimitMiddleware,
        path=f"{resolved_settings.api_prefix}/transcriptions",
        max_body_bytes=resolved_settings.max_upload_bytes + 1024 * 1024,
        max_upload_bytes=resolved_settings.max_upload_bytes,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        path=f"{resolved_settings.api_prefix}/conversations",
        max_body_bytes=resolved_settings.max_upload_bytes + 1024 * 1024,
        max_upload_bytes=resolved_settings.max_upload_bytes,
    )
    app.add_middleware(
        ObservabilityMiddleware,
        observability=resolved_observability,
    )
    app.add_middleware(
        RequestIdMiddleware,
        header_name=resolved_settings.request_id_header,
    )
    if resolved_settings.security_headers_enabled:
        app.add_middleware(
            SecurityHeadersMiddleware,
            hsts_max_age_seconds=resolved_settings.hsts_max_age_seconds,
        )
    install_exception_handlers(app)
    app.include_router(health.router)
    if resolved_settings.metrics_enabled:
        app.include_router(metrics.router)
    app.include_router(live.router, prefix=resolved_settings.api_prefix)
    app.include_router(
        v1.router,
        prefix=resolved_settings.api_prefix,
        dependencies=[Depends(authenticate_api_key)],
        responses={
            status.HTTP_429_TOO_MANY_REQUESTS: {
                "model": ProblemDetail,
                "description": "Caller rate limit exceeded.",
            }
        },
    )

    if resolved_settings.service_version != __version__:
        raise ValueError("configured service version must match the package version")
    return app


def _export_service(settings: Settings) -> TranscriptExportService:
    """Plain text always renders; PDF joins it only when fpdf2 is installed.

    The renderer imports fpdf lazily so it stays testable without the optional
    package, which means availability has to be probed here instead. Registering
    it blindly would report a missing font when the real problem is a missing
    dependency.
    """

    renderers: dict[ExportFormat, DocumentRenderer] = {
        ExportFormat.TXT: PlainTextDocumentRenderer(),
    }
    if importlib.util.find_spec("fpdf") is None:
        logger.warning("PDF export is unavailable because fpdf2 is not installed")
        return TranscriptExportService(renderers)

    from speech_intelligence_api.adapters.pdf_documents import PdfDocumentRenderer

    renderers[ExportFormat.PDF] = PdfDocumentRenderer(settings)
    return TranscriptExportService(renderers)
