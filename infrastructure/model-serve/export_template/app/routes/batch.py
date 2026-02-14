"""
Batch prediction route for standalone model serving.

Reads data from a configured source (database or file),
runs predictions, and optionally writes results back.
"""

import time
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from ..schemas import BatchPredictRequest, BatchPredictResponse, ErrorResponse
from ..config import DataSourceConfig, DataOutputConfig
from ..data_source import read_batch_data
from ..data_sink import write_predictions
from ..prediction_utils import predict_with_confidence

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        400: {"model": ErrorResponse, "description": "Data source error"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Batch prediction",
    description="Read data from configured source, predict, and optionally write results back."
)
async def predict_batch(
    app_request: Request,
    request: Optional[BatchPredictRequest] = None
):
    model = getattr(app_request.app.state, "model", None)
    metadata = getattr(app_request.app.state, "metadata", {})
    config = getattr(app_request.app.state, "config", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_loaded", "detail": "No model is currently loaded"}
        )

    if config is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "config_missing", "detail": "Application config not loaded"}
        )

    # Determine source and output configs (allow runtime overrides)
    if request and request.source_override:
        source_config = DataSourceConfig(**request.source_override)
    else:
        source_config = config.data_source

    if request and request.output_override:
        output_config = DataOutputConfig(**request.output_override)
    else:
        output_config = config.data_output

    # Read input data
    try:
        df = read_batch_data(source_config)
        logger.info(f"Batch data loaded: {len(df)} rows, {len(df.columns)} columns")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "data_source_error", "detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Failed to read batch data: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "data_source_error", "detail": str(e)}
        )

    # Run predictions
    try:
        start_time = time.time()
        result = predict_with_confidence(
            model, df, task_type=metadata.get("task_type")
        )
        inference_time = (time.time() - start_time) * 1000

        predictions_list = result["predictions"]
        confidence_scores = result["confidence_scores"]
        class_probabilities = result["class_probabilities"]

        logger.info(
            f"Batch prediction completed: count={len(predictions_list)}, "
            f"time={inference_time:.2f}ms"
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "prediction_error", "detail": str(e)}
        )

    # Write output
    output_destinations = []
    output_types = [t.strip() for t in output_config.type.split("+")]

    if any(t in output_types for t in ("database", "csv", "json")):
        try:
            result_df = df.copy()
            result_df["prediction"] = predictions_list
            if confidence_scores is not None:
                result_df["confidence"] = confidence_scores
            write_result = write_predictions(result_df, output_config)
            output_destinations.extend(write_result.get("destinations", []))
        except Exception as e:
            logger.error(f"Failed to write predictions: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": "output_error", "detail": str(e)}
            )

    # Build response
    include_predictions = "response" in output_types
    if include_predictions:
        output_destinations.append("response")

    return BatchPredictResponse(
        model_name=metadata.get("name", "unknown"),
        model_version=metadata.get("version", "unknown"),
        prediction_count=len(predictions_list),
        inference_time_ms=round(inference_time, 2),
        output_destinations=output_destinations,
        predictions=predictions_list if include_predictions else None,
        confidence_scores=confidence_scores if include_predictions else None,
        class_probabilities=class_probabilities if include_predictions else None,
    )
