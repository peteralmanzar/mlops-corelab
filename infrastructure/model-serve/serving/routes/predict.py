"""
Prediction routes for FastAPI model serving.
"""

import time
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import PredictRequest, PredictResponse, ErrorResponse
from ..model_manager import ModelManager
from ..dependencies import get_model_manager
from ..prediction_utils import predict_with_confidence

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict/{model_name}",
    response_model=PredictResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Make prediction",
    description="Make predictions using a loaded champion model."
)
async def predict(
    model_name: str,
    request: PredictRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Make predictions using a specific model.

    Args:
        model_name: Name of the model to use (must be loaded as champion)
        request: Prediction request with input data

    Returns:
        Prediction response with results
    """
    # Get the model
    loaded_model = manager.get_model(model_name)

    if loaded_model is None:
        available = [m['name'] for m in manager.list_models()]
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "detail": f"Model '{model_name}' not found. Available models: {available}",
                "model_name": model_name
            }
        )

    try:
        df = pd.DataFrame(request.data)

        start_time = time.time()
        result = predict_with_confidence(
            loaded_model.model, df, task_type=loaded_model.task_type
        )
        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction completed: model={model_name}, "
            f"count={len(result['predictions'])}, time={inference_time:.2f}ms"
        )

        return PredictResponse(
            model_name=loaded_model.name,
            model_version=loaded_model.version,
            predictions=result["predictions"],
            prediction_count=len(result["predictions"]),
            inference_time_ms=round(inference_time, 2),
            confidence_scores=result["confidence_scores"],
            class_probabilities=result["class_probabilities"],
        )

    except Exception as e:
        logger.error(f"Prediction error for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_error",
                "detail": str(e),
                "model_name": model_name
            }
        )
