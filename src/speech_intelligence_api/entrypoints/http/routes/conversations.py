"""Asynchronous multi-speaker conversation endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Request, Response, UploadFile, status

from speech_intelligence_api.application.conversations import (
    ConversationSubmissionService,
    ConversationUploadCommand,
)
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.enums import (
    ChineseScript,
    LanguageCode,
    LanguageSelectionMode,
)
from speech_intelligence_api.domain.errors import (
    DependencyUnavailableError,
    ErrorCode,
    ServiceError,
)
from speech_intelligence_api.domain.models import LanguageSelection
from speech_intelligence_api.entrypoints.http.schemas import JobAcceptedResponse, ProblemDetail

router = APIRouter(tags=["conversations"])


@router.post(
    "/conversations",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ProblemDetail},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ProblemDetail},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ProblemDetail},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ProblemDetail},
    },
    operation_id="create_conversation",
    summary="Queue native-script multi-speaker conversation transcription",
)
async def create_conversation(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="Supported private conversation audio.")],
    language_mode: Annotated[LanguageSelectionMode, Form()] = LanguageSelectionMode.EXPLICIT,
    language: Annotated[LanguageCode | None, Form()] = LanguageCode.ENGLISH,
    chinese_script: Annotated[ChineseScript | None, Form()] = None,
    expected_speakers: Annotated[int | None, Form(ge=2, le=100)] = None,
    vocabulary: Annotated[list[str] | None, Form()] = None,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=16,
            max_length=128,
            description="Optional retry-safe key for conversation submission.",
        ),
    ] = None,
) -> JobAcceptedResponse:
    """Validate, normalize, enqueue, and immediately remove the raw upload."""

    settings: Settings = request.app.state.settings
    service: ConversationSubmissionService | None = request.app.state.conversation_service
    if service is None:
        raise DependencyUnavailableError
    try:
        try:
            selected_language = (
                None if language_mode is LanguageSelectionMode.AUTOMATIC else language
            )
            selected_chinese_script = (
                chinese_script if selected_language in {None, LanguageCode.CHINESE} else None
            )
            command = ConversationUploadCommand(
                filename=file.filename or "",
                declared_media_type=file.content_type or "",
                chunks=_upload_chunks(file, settings.upload_chunk_bytes),
                language=LanguageSelection(
                    mode=language_mode,
                    language=selected_language,
                    chinese_script=selected_chinese_script,
                ),
                expected_speakers=expected_speakers,
                vocabulary=tuple(vocabulary or ()),
                idempotency_key=idempotency_key,
            )
        except ValueError:
            raise ServiceError(
                ErrorCode.INVALID_REQUEST,
                "The conversation options are invalid.",
            ) from None
        outcome = await service.execute(command)
    finally:
        await file.close()

    response.status_code = status.HTTP_202_ACCEPTED
    job_path = f"{settings.api_prefix}/jobs/{outcome.job.job_id}"
    return JobAcceptedResponse(
        task="diarize",
        request_id=str(request.state.request_id),
        job_id=outcome.job.job_id,
        kind=outcome.job.kind,
        status=outcome.job.status,
        progress_percent=outcome.job.progress_percent,
        created_at=outcome.job.created_at.isoformat(),
        expires_at=outcome.job.expires_at.isoformat(),
        replayed=outcome.replayed,
        links={
            "self": job_path,
            "result": f"{job_path}/result",
            "cancel": f"{job_path}/cancel",
        },
    )


async def _upload_chunks(file: UploadFile, chunk_bytes: int) -> AsyncIterator[bytes]:
    while chunk := await file.read(chunk_bytes):
        yield chunk
