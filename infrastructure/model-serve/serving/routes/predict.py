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
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)

        # Time the prediction
        start_time = time.time()

        # Make prediction
        predictions = loaded_model.predict(df)

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Convert predictions to list
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)

        logger.info(
            f"Prediction completed: model={model_name}, "
            f"count={len(predictions_list)}, time={inference_time:.2f}ms"
        )

        return PredictResponse(
            model_name=loaded_model.name,
            model_version=loaded_model.version,
            predictions=predictions_list,
            prediction_count=len(predictions_list),
            inference_time_ms=round(inference_time, 2)
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
