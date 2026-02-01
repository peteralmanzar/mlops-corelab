"""
Admin routes for FastAPI model serving.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import ModelsListResponse, ModelInfo, ReloadResponse, ErrorResponse
from ..model_manager import ModelManager
from ..dependencies import get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List loaded models",
    description="List all currently loaded champion models."
)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """List all loaded models with their metadata."""
    models = manager.list_models()

    return ModelsListResponse(
        count=len(models),
        models=[
            ModelInfo(
                name=m['name'],
                version=m['version'],
                run_id=m['run_id'],
                model_type=m['model_type'],
                loaded_at=m['loaded_at']
            )
            for m in models
        ]
    )


@router.post(
    "/reload",
    response_model=ReloadResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Reload error"}
    },
    summary="Reload models",
    description="Reload all champion models from MLflow. Use after promoting a new model."
)
async def reload_models(manager: ModelManager = Depends(get_model_manager)):
    """
    Reload all champion models from MLflow.

    This endpoint triggers a hot-reload of all models, discovering any
    newly promoted champions and unloading models that are no longer champions.
    """
    try:
        logger.info("Model reload requested via API")
        result = manager.reload_models()

        logger.info(
            f"Reload complete: {result['previous_count']} -> {result['current_count']} models. "
            f"Added: {result['added']}, Removed: {result['removed']}, Updated: {result['updated']}"
        )

        return ReloadResponse(**result)

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "reload_error",
                "detail": str(e)
            }
        )
