"""Authenticated Prometheus exposition endpoint for operators."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from speech_intelligence_api.entrypoints.http.dependencies import authenticate_api_key_only
from speech_intelligence_api.entrypoints.http.security import ApiPrincipal
from speech_intelligence_api.ports.observability import Observability

router = APIRouter(tags=["operations"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(
    request: Request,
    _: Annotated[ApiPrincipal, Depends(authenticate_api_key_only)],
) -> Response:
    """Render privacy-safe, low-cardinality process metrics."""

    observability: Observability = request.app.state.observability
    payload, content_type = observability.render_metrics()
    return Response(
        content=payload,
        headers={
            "Cache-Control": "no-store",
            "Content-Type": content_type or CONTENT_TYPE_LATEST,
        },
    )
