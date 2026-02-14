"""
Admin routes for FastAPI model serving.
"""

import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..schemas import ModelsListResponse, ModelInfo, ReloadResponse, ErrorResponse, ModelFeaturesResponse
from ..model_manager import ModelManager
from ..dependencies import get_model_manager
from ..export_service import ModelExportService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List loaded models",
    description="List all currently loaded champion models with feature summary."
)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """List all loaded models with their metadata and feature summary."""
    models = manager.list_models()

    return ModelsListResponse(
        count=len(models),
        models=[
            ModelInfo(
                name=m['name'],
                version=m['version'],
                run_id=m['run_id'],
                model_type=m['model_type'],
                loaded_at=m['loaded_at'],
                feature_count=m.get('feature_count'),
                feature_metadata_available=m.get('feature_metadata_available', False)
            )
            for m in models
        ]
    )


@router.get(
    "/models/{model_name}/features",
    response_model=ModelFeaturesResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"}
    },
    summary="Get model input features",
    description="Get detailed input feature information for a specific loaded model."
)
async def get_model_features(
    model_name: str,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get detailed input feature information for a specific model.

    Returns the feature names, data types, and count that the model expects
    as input. This information is retrieved from MLflow artifacts logged
    during model training.
    """
    features = manager.get_model_features(model_name)

    if features is None:
        available = [m['name'] for m in manager.list_models()]
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "detail": f"Model '{model_name}' not found. Available models: {available}",
                "model_name": model_name
            }
        )

    return ModelFeaturesResponse(**features)


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


def _cleanup_export_temp(zip_path: Path):
    """Clean up temporary export files after download completes."""
    try:
        parent = zip_path.parent
        shutil.rmtree(parent, ignore_errors=True)
        logger.info(f"Cleaned up export temp dir: {parent}")
    except Exception as e:
        logger.warning(f"Failed to clean up export temp: {e}")


@router.post(
    "/export/{model_name}",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Export error"}
    },
    summary="Export model as standalone Docker package",
    description="Download a zip containing a self-contained FastAPI Docker project for the specified model."
)
async def export_model(
    model_name: str,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Export a loaded champion model as a standalone Docker package.

    The zip contains a Dockerfile, FastAPI app, model artifacts, and config
    template — ready to `docker build` and `docker run`.
    """
    export_service = ModelExportService(manager)

    try:
        logger.info(f"Export requested for model: {model_name}")
        zip_path = export_service.export_model(model_name)

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"{model_name}_standalone.zip",
            background=BackgroundTask(_cleanup_export_temp, zip_path)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "detail": str(e),
                "model_name": model_name
            }
        )
    except Exception as e:
        logger.error(f"Export failed for {model_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "export_error",
                "detail": str(e),
                "model_name": model_name
            }
        )
