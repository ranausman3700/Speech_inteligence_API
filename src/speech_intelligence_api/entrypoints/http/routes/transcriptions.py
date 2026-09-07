"""Synchronous batch-transcription endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Request, Response, UploadFile, status

from speech_intelligence_api.application.transcriptions import (
    BatchTranscriptionService,
    QueuedTranscriptionOutcome,
    UploadCommand,
)
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.enums import (
    ChineseScript,
    LanguageCode,
    LanguageSelectionMode,
    ProcessingMode,
)
from speech_intelligence_api.domain.errors import ErrorCode, ServiceError
from speech_intelligence_api.domain.models import LanguageSelection
from speech_intelligence_api.entrypoints.http.schemas import (
    JobAcceptedResponse,
    ProblemDetail,
    TranscriptionResponse,
    TranscriptSegmentResponse,
    TranscriptWordResponse,
)

router = APIRouter(tags=["transcriptions"])


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse | JobAcceptedResponse,
    responses={
        status.HTTP_202_ACCEPTED: {"model": JobAcceptedResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ProblemDetail},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ProblemDetail},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ProblemDetail},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ProblemDetail},
    },
    operation_id="create_transcription",
    summary="Transcribe a short audio recording in its native script",
)
async def create_transcription(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="Supported private audio upload.")],
    language_mode: Annotated[LanguageSelectionMode, Form()] = LanguageSelectionMode.EXPLICIT,
    language: Annotated[LanguageCode | None, Form()] = LanguageCode.ENGLISH,
    chinese_script: Annotated[ChineseScript | None, Form()] = None,
    processing_mode: Annotated[ProcessingMode, Form()] = ProcessingMode.AUTO,
    vocabulary: Annotated[list[str] | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=16,
            max_length=128,
            description="Optional retry-safe key for asynchronous submission.",
        ),
    ] = None,
) -> TranscriptionResponse | JobAcceptedResponse:
    """Stream, validate, normalize, transcribe, and immediately delete uploaded audio."""

    settings: Settings = request.app.state.settings
    service: BatchTranscriptionService = request.app.state.transcription_service
    try:
        try:
            selected_language = (
                None if language_mode is LanguageSelectionMode.AUTOMATIC else language
            )
            selected_chinese_script = (
                chinese_script if selected_language in {None, LanguageCode.CHINESE} else None
            )
            selection = LanguageSelection(
                mode=language_mode,
                language=selected_language,
                chinese_script=selected_chinese_script,
            )
            command = UploadCommand(
                filename=file.filename or "",
                declared_media_type=file.content_type or "",
                chunks=_upload_chunks(file, settings.upload_chunk_bytes),
                language=selection,
                processing_mode=processing_mode,
                vocabulary=tuple(vocabulary or ()),
                word_timestamps=word_timestamps,
                idempotency_key=idempotency_key,
            )
        except ValueError:
            raise ServiceError(
                ErrorCode.INVALID_REQUEST,
                "The transcription options are invalid.",
            ) from None
        outcome = await service.execute(command)
    finally:
        await file.close()

    if isinstance(outcome, QueuedTranscriptionOutcome):
        response.status_code = status.HTTP_202_ACCEPTED
        job_path = f"{settings.api_prefix}/jobs/{outcome.job.job_id}"
        return JobAcceptedResponse(
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

    return TranscriptionResponse(
        request_id=str(request.state.request_id),
        language=outcome.result.language,
        language_confidence_estimate=outcome.result.language_confidence_estimate,
        chinese_script=outcome.result.chinese_script,
        duration_seconds=outcome.duration_seconds,
        text=outcome.result.text,
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
            for segment in outcome.result.segments
        ],
    )


async def _upload_chunks(file: UploadFile, chunk_bytes: int) -> AsyncIterator[bytes]:
    while chunk := await file.read(chunk_bytes):
        yield chunk
