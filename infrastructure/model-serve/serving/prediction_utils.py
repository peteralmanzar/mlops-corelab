"""
Prediction utilities with confidence score extraction.

Handles sklearn Pipelines (including those with KerasModelWrapper as the last
step), standard sklearn classifiers, and raw Keras models.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def predict_with_confidence(
    model,
    data,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make predictions and extract confidence scores when applicable.

    Args:
        model: sklearn Pipeline, sklearn estimator, or Keras model.
        data: Input data (DataFrame or array).
        task_type: One of 'regression', 'binary_classification',
            'multi_classification'. If None, inferred from output shape.

    Returns:
        Dict with keys:
            predictions     – list of predicted labels or values
            confidence_scores – list of float [0, 1] or None (regression)
            class_probabilities – list of per-class probability arrays or None
    """
    raw_output = _get_model_output(model, data)
    raw = np.asarray(raw_output, dtype=float)

    # Flatten trailing (n, 1) → (n,)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw.ravel()

    # Infer task type from output when not supplied
    if task_type is None:
        task_type = _infer_task_type(raw)

    if task_type == "multi_classification" and raw.ndim == 2 and raw.shape[1] > 1:
        predictions = raw.argmax(axis=1).tolist()
        confidence = raw.max(axis=1).tolist()
        probabilities = raw.tolist()
        return {
            "predictions": predictions,
            "confidence_scores": [round(c, 6) for c in confidence],
            "class_probabilities": [
                [round(p, 6) for p in row] for row in probabilities
            ],
        }

    if task_type == "binary_classification":
        raw_flat = raw.ravel() if raw.ndim > 1 else raw
        predictions = (raw_flat >= 0.5).astype(int).tolist()
        confidence = np.where(raw_flat >= 0.5, raw_flat, 1.0 - raw_flat).tolist()
        return {
            "predictions": predictions,
            "confidence_scores": [round(c, 6) for c in confidence],
            "class_probabilities": None,
        }

    # Regression or unknown
    predictions = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    return {
        "predictions": predictions,
        "confidence_scores": None,
        "class_probabilities": None,
    }


def _get_model_output(model, data):
    """Get raw model output, trying predict() then transform()."""
    try:
        return model.predict(data)
    except (AttributeError, TypeError):
        pass

    try:
        return model.transform(data)
    except (AttributeError, TypeError):
        pass

    raise RuntimeError("Model does not support predict() or transform()")


def _infer_task_type(raw: np.ndarray) -> str:
    """Best-effort inference of task type from model output values."""
    if raw.ndim == 2 and raw.shape[1] > 1:
        return "multi_classification"

    flat = raw.ravel()
    if flat.size > 0:
        min_val, max_val = float(flat.min()), float(flat.max())
        if 0.0 <= min_val and max_val <= 1.0:
            return "binary_classification"

    return "regression"
