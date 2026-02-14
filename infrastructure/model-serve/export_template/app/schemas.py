"""
Pydantic schemas for standalone model serving request/response models.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for ad-hoc prediction endpoint."""
    data: List[Dict[str, Any]] = Field(
        ...,
        description="List of records to predict. Each record is a dict of feature name -> value.",
    )


class PredictResponse(BaseModel):
    """Response schema for ad-hoc prediction endpoint."""
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    predictions: List[Any] = Field(..., description="Prediction results (class labels for classification, values for regression)")
    prediction_count: int = Field(..., description="Number of predictions made")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    confidence_scores: Optional[List[float]] = Field(
        None,
        description="Confidence score per prediction (0-1). Present for classification models only."
    )
    class_probabilities: Optional[List[List[float]]] = Field(
        None,
        description="Per-class probability arrays. Present for multi-class classification only."
    )


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    source_override: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional runtime override for data source config"
    )
    output_override: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional runtime override for output config"
    )


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    prediction_count: int = Field(..., description="Number of predictions made")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    output_destinations: List[str] = Field(
        default_factory=list,
        description="Where predictions were written (e.g. ['database', 'response'])"
    )
    predictions: Optional[List[Any]] = Field(
        None,
        description="Prediction results (included only if output type includes 'response')"
    )
    confidence_scores: Optional[List[float]] = Field(
        None,
        description="Confidence score per prediction (0-1). Present for classification models only."
    )
    class_probabilities: Optional[List[List[float]]] = Field(
        None,
        description="Per-class probability arrays. Present for multi-class classification only."
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReadinessResponse(BaseModel):
    """Response schema for readiness check endpoint."""
    status: str = Field(..., description="Readiness status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
