"""
Ad-hoc prediction route for standalone model serving.
"""

import time
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from ..schemas import PredictRequest, PredictResponse, ErrorResponse
from ..prediction_utils import predict_with_confidence

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Make prediction",
    description="Make predictions by sending data in the request body."
)
async def predict(request: PredictRequest, app_request: Request):
    model = getattr(app_request.app.state, "model", None)
    metadata = getattr(app_request.app.state, "metadata", {})

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_loaded", "detail": "No model is currently loaded"}
        )

    try:
        df = pd.DataFrame(request.data)

        start_time = time.time()
        result = predict_with_confidence(
            model, df, task_type=metadata.get("task_type")
        )
        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction completed: count={len(result['predictions'])}, "
            f"time={inference_time:.2f}ms"
        )

        return PredictResponse(
            model_name=metadata.get("name", "unknown"),
            model_version=metadata.get("version", "unknown"),
            predictions=result["predictions"],
            prediction_count=len(result["predictions"]),
            inference_time_ms=round(inference_time, 2),
            confidence_scores=result["confidence_scores"],
            class_probabilities=result["class_probabilities"],
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "prediction_error", "detail": str(e)}
        )
