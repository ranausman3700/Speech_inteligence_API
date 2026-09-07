"""Unauthenticated orchestrator health endpoints."""

from fastapi import APIRouter, Request, Response, status

from speech_intelligence_api.application.readiness import ReadinessService
from speech_intelligence_api.config import Settings
from speech_intelligence_api.entrypoints.http.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    operation_id="get_liveness",
    summary="Process liveness",
)
async def liveness(request: Request) -> HealthResponse:
    """Confirm that the API process can serve requests."""

    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.service_version,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
    operation_id="get_readiness",
    summary="Dependency readiness",
)
async def readiness(request: Request, response: Response) -> HealthResponse:
    """Report whether the API and configured dependencies can accept traffic."""

    settings: Settings = request.app.state.settings
    readiness_service: ReadinessService = request.app.state.readiness_service
    report = await readiness_service.evaluate()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if report.ready else "unavailable",
        service=settings.service_name,
        version=settings.service_version,
        checks=report.checks,
    )
