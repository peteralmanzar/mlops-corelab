"""
Health check routes for standalone model serving.
"""

from datetime import datetime
from fastapi import APIRouter, Request

from ..schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Basic liveness check."
)
async def health_check():
    return HealthResponse(status="healthy", timestamp=datetime.now())


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Check if the model is loaded and ready to serve."
)
async def readiness_check(request: Request):
    model = getattr(request.app.state, "model", None)
    metadata = getattr(request.app.state, "metadata", {})

    if model is not None:
        return ReadinessResponse(
            status="ready",
            model_loaded=True,
            model_name=metadata.get("name")
        )
    return ReadinessResponse(
        status="not_ready",
        model_loaded=False
    )
