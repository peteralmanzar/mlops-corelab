"""
Model loader for standalone serving.

Loads a serialized sklearn Pipeline from disk using joblib.
No MLflow dependency required.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))


def load_model_from_disk() -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and metadata from the model directory.

    Returns:
        Tuple of (model, metadata_dict)
    """
    model_path = MODEL_DIR / "model.pkl"
    metadata_path = MODEL_DIR / "model_metadata.json"

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Model loaded: {type(model).__name__}")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(
            f"Model metadata loaded: name={metadata.get('name')}, "
            f"version={metadata.get('version')}, "
            f"features={metadata.get('feature_count', 'unknown')}"
        )
    else:
        logger.warning("No model_metadata.json found")

    return model, metadata
