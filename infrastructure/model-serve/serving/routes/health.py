"""
Health check routes for FastAPI model serving.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from ..schemas import HealthResponse, ReadinessResponse
from ..model_manager import ModelManager
from ..dependencies import get_model_manager

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Basic health check to verify the service is running."
)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Check if the service is ready to serve predictions (models loaded)."
)
async def readiness_check(manager: ModelManager = Depends(get_model_manager)):
    """Check if models are loaded and ready to serve."""
    models = manager.list_models()
    is_ready = manager.is_ready()

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        models_loaded=len(models),
        models=[m['name'] for m in models]
    )
