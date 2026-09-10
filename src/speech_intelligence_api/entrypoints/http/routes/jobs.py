"""Ephemeral asynchronous job endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Request, Response, status

from speech_intelligence_api.application.jobs import JobManagementService
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.jobs import (
    ConversationJobResult,
    TranscriptionJobResult,
)
from speech_intelligence_api.domain.models import JobRecord
from speech_intelligence_api.entrypoints.http.schemas import (
    ConversationSegmentResponse,
    JobConversationResultResponse,
    JobStatusResponse,
    JobTranscriptionResultResponse,
    ProblemDetail,
    TranscriptSegmentResponse,
    TranscriptWordResponse,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])
JobId = Annotated[str, Path(pattern=r"^job_[0-9a-f]{32}$")]

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ProblemDetail},
    status.HTTP_409_CONFLICT: {"model": ProblemDetail},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ProblemDetail},
}


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    responses=_ERROR_RESPONSES,
    operation_id="get_job",
    summary="Get asynchronous job status",
)
async def get_job(request: Request, job_id: JobId) -> JobStatusResponse:
    service = _service(request)
    return _status_response(request, await service.get(job_id))


@router.get(
    "/{job_id}/result",
    response_model=JobTranscriptionResultResponse | JobConversationResultResponse,
    responses=_ERROR_RESPONSES,
    operation_id="get_job_result",
    summary="Get a completed transcription result",
)
async def get_job_result(
    request: Request,
    job_id: JobId,
) -> JobTranscriptionResultResponse | JobConversationResultResponse:
    outcome = await _service(request).result(job_id)
    if isinstance(outcome, ConversationJobResult):
        conversation = outcome.result
        return JobConversationResultResponse(
            request_id=str(request.state.request_id),
            job_id=job_id,
            language=conversation.language,
            language_confidence_estimate=conversation.language_confidence_estimate,
            chinese_script=conversation.chinese_script,
            duration_seconds=outcome.duration_seconds,
            raw_transcript=conversation.raw_transcript,
            speaker_count=conversation.speaker_count,
            formatted_transcript=conversation.formatted_transcript,
            segments=[_conversation_segment_response(segment) for segment in conversation.segments],
        )
    if not isinstance(outcome, TranscriptionJobResult):
        raise TypeError("job result kind is not supported")
    result = outcome.result
    return JobTranscriptionResultResponse(
        request_id=str(request.state.request_id),
        job_id=job_id,
        language=result.language,
        language_confidence_estimate=result.language_confidence_estimate,
        chinese_script=result.chinese_script,
        duration_seconds=outcome.duration_seconds,
        text=result.text,
        segments=[_segment_response(segment) for segment in result.segments],
    )


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatusResponse,
    responses=_ERROR_RESPONSES,
    operation_id="cancel_job",
    summary="Request cooperative job cancellation",
)
async def cancel_job(request: Request, job_id: JobId) -> JobStatusResponse:
    return _status_response(request, await _service(request).cancel(job_id))


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_ERROR_RESPONSES,
    operation_id="delete_job",
    summary="Delete terminal ephemeral job state",
)
async def delete_job(request: Request, job_id: JobId) -> Response:
    await _service(request).delete(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _service(request: Request) -> JobManagementService:
    service: JobManagementService | None = request.app.state.job_service
    if service is None:
        from speech_intelligence_api.domain.errors import DependencyUnavailableError

        raise DependencyUnavailableError
    return service


def _status_response(request: Request, record: JobRecord) -> JobStatusResponse:
    settings: Settings = request.app.state.settings
    job_path = f"{settings.api_prefix}/jobs/{record.job_id}"
    return JobStatusResponse(
        request_id=str(request.state.request_id),
        job_id=record.job_id,
        kind=record.kind,
        status=record.status,
        progress_percent=record.progress_percent,
        cancellation_requested=record.cancellation_requested,
        failure_code=record.failure_code,
        created_at=record.created_at.isoformat(),
        expires_at=record.expires_at.isoformat(),
        links={"self": job_path, "result": f"{job_path}/result"},
    )


def _segment_response(segment: object) -> TranscriptSegmentResponse:
    from speech_intelligence_api.domain.models import TranscriptSegment

    if not isinstance(segment, TranscriptSegment):
        raise TypeError("job result contains an invalid segment")
    return TranscriptSegmentResponse(
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


def _conversation_segment_response(segment: object) -> ConversationSegmentResponse:
    from speech_intelligence_api.domain.models import TranscriptSegment

    if not isinstance(segment, TranscriptSegment):
        raise TypeError("conversation result contains an invalid segment")
    transcript = _segment_response(segment)
    return ConversationSegmentResponse(
        **transcript.model_dump(),
        speaker=segment.speaker,
        speaker_uncertain=segment.speaker_uncertain,
        speaker_confidence_estimate=segment.speaker_confidence_estimate,
        overlapping_speech=segment.overlapping_speech,
    )
