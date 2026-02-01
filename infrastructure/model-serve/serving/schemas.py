"""
Pydantic schemas for FastAPI request/response models.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    data: List[Dict[str, Any]] = Field(
        ...,
        description="List of records to predict. Each record is a dict of feature name -> value.",
        example=[{"Pclass": 1, "Sex": "male", "Age": 35, "Fare": 100.0}]
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    predictions: List[Any] = Field(..., description="Prediction results")
    prediction_count: int = Field(..., description="Number of predictions made")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Health status", example="healthy")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReadinessResponse(BaseModel):
    """Response schema for readiness check endpoint."""
    status: str = Field(..., description="Readiness status")
    models_loaded: int = Field(..., description="Number of models loaded")
    models: List[str] = Field(default_factory=list, description="List of loaded model names")


class ModelInfo(BaseModel):
    """Schema for model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    run_id: str = Field(..., description="MLflow run ID")
    model_type: str = Field(..., description="Type of model (sklearn_pipeline, keras, etc.)")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")


class ModelsListResponse(BaseModel):
    """Response schema for listing loaded models."""
    count: int = Field(..., description="Number of loaded models")
    models: List[ModelInfo] = Field(default_factory=list, description="List of model info")


class ReloadResponse(BaseModel):
    """Response schema for reload endpoint."""
    status: str = Field(..., description="Reload status")
    previous_count: int = Field(..., description="Number of models before reload")
    current_count: int = Field(..., description="Number of models after reload")
    added: List[str] = Field(default_factory=list, description="Models added")
    removed: List[str] = Field(default_factory=list, description="Models removed")
    updated: List[str] = Field(default_factory=list, description="Models with version changes")
    models: Dict[str, str] = Field(default_factory=dict, description="Model name -> version mapping")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    model_name: Optional[str] = Field(None, description="Model name if applicable")
