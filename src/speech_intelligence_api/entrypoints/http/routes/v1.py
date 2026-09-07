"""Version 1 API contracts available before inference adapters are installed."""

from fastapi import APIRouter, Request, status

from speech_intelligence_api.domain.enums import SUPPORTED_LANGUAGE_VARIANTS
from speech_intelligence_api.entrypoints.http.routes import conversations, jobs, transcriptions
from speech_intelligence_api.entrypoints.http.schemas import (
    CapabilitiesResponse,
    ConversationCapability,
    LanguageCapability,
    LiveTranscriptionCapability,
    ProblemDetail,
)

router = APIRouter()
router.include_router(transcriptions.router)
router.include_router(conversations.router)
router.include_router(jobs.router)


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ProblemDetail,
            "description": "Missing or invalid API key.",
        }
    },
    tags=["capabilities"],
    operation_id="get_capabilities",
    summary="List supported transcription languages",
)
async def capabilities(request: Request) -> CapabilitiesResponse:
    """Return the immutable native-script language allowlist."""

    settings = request.app.state.settings
    live_capability = (
        LiveTranscriptionCapability(
            websocket_path=f"{settings.api_prefix}/live-transcription",
        )
        if settings.live_transcription_enabled
        else None
    )
    return CapabilitiesResponse(
        languages=[
            LanguageCapability(
                name=variant.name,
                code=variant.code,
                chinese_script=variant.chinese_script,
            )
            for variant in SUPPORTED_LANGUAGE_VARIANTS
        ],
        live_transcription=live_capability,
        conversations=(
            ConversationCapability(
                endpoint=f"{settings.api_prefix}/conversations",
                expected_speakers_maximum=settings.diarization_max_expected_speakers,
            )
            if settings.diarization_enabled and settings.async_jobs_enabled
            else None
        ),
    )
