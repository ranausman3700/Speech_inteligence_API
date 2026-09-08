"""Transcript export endpoint.

Export is stateless on purpose: the caller sends back text it already holds, so
one endpoint serves synchronous transcriptions, asynchronous job results and live
dictation alike, and no transcript is written to storage to make a download work.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Request, Response, status
from pydantic import Field

from speech_intelligence_api.application.exports import ExportCommand, TranscriptExportService
from speech_intelligence_api.domain.enums import ExportContent, ExportFormat
from speech_intelligence_api.domain.errors import ErrorCode, ServiceError
from speech_intelligence_api.entrypoints.http.schemas import ApiSchema, ProblemDetail

router = APIRouter(tags=["exports"])


class ExportRequest(ApiSchema):
    """A transcript plus the view and format the caller wants to download."""

    transcript: Annotated[str, Field(min_length=1, max_length=200_000)]
    content: ExportContent = ExportContent.TRANSCRIPT
    format: ExportFormat = ExportFormat.TXT
    title: Annotated[str | None, Field(default=None, max_length=120)] = None


@router.post(
    "/exports",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "text/plain": {"schema": {"type": "string", "format": "binary"}},
                "application/pdf": {"schema": {"type": "string", "format": "binary"}},
            },
            "description": "The rendered document as a file attachment.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ProblemDetail},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ProblemDetail},
    },
    operation_id="create_export",
    summary="Download a transcript or its summary as txt or pdf",
)
async def create_export(request: Request, payload: ExportRequest) -> Response:
    """Render the requested view and return it as a download."""

    service: TranscriptExportService = request.app.state.export_service
    try:
        command = ExportCommand(
            transcript=payload.transcript,
            content=payload.content,
            export_format=payload.format,
            title=payload.title,
        )
    except ValueError:
        raise ServiceError(
            ErrorCode.INVALID_REQUEST,
            "The export options are invalid.",
        ) from None

    document = service.execute(command)
    return Response(
        content=document.content,
        media_type=document.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{document.filename}"',
            "Cache-Control": "no-store",
        },
    )
