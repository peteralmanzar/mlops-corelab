"""Preprocessing pipeline factory and persistence helpers.

Builds sklearn-style pipelines from a simple spec and provides save/load helpers.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
from typing import Dict, Any, List, Optional

from config_load import Config
from data_transform import (
        PipelineOneHotEncoder,
        PipelineFeatureStandardScaler,
        PipelineFeatureDropper,
        PipelineDateSpliter,
        PipelineIndexSetter,
        PipelineSlidingWindow,
        PipelineImputer,
        PipelineNullRowDropper,
    )

logger = logging.getLogger(__name__)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a file, auto-detecting format based on extension.

    Supports:
        - .csv: Comma-separated values
        - .json: JSON (records orientation by default, or auto-detect)

    Args:
        file_path: Path to the data file

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file extension is not supported
    """
    ext = Path(file_path).suffix.lower()

    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .csv, .json")


def _get_pipeline_spec(config: Config) -> Dict[str, Any]:
    """Extract pipeline specification from config.
    
    Supports both nested PREPROCESSING.PIPELINE_SPEC and flat PREPROCESSING_PIPELINE_SPEC.
    
    Parameters
    ----------
    config : Config
        Configuration object.
    
    Returns
    -------
    dict
        Pipeline specification with 'steps' key.
    """
    # Try nested structure first (PREPROCESSING.PIPELINE_SPEC)
    if hasattr(config, 'PREPROCESSING') and isinstance(config.PREPROCESSING, dict):
        spec = config.PREPROCESSING.get('PIPELINE_SPEC')
        if spec and isinstance(spec, dict) and 'steps' in spec:
            logger.info("Loaded pipeline spec from PREPROCESSING.PIPELINE_SPEC")
            return spec
    
    # Try flat structure (PREPROCESSING_PIPELINE_SPEC)
    if hasattr(config, 'PREPROCESSING_PIPELINE_SPEC') and isinstance(config.PREPROCESSING_PIPELINE_SPEC, dict):
        spec = config.PREPROCESSING_PIPELINE_SPEC
        if 'steps' in spec:
            logger.info("Loaded pipeline spec from PREPROCESSING_PIPELINE_SPEC")
            return spec
    
    # Default empty pipeline
    logger.warning("No pipeline specification found in config. Using empty pipeline.")
    return {"steps": []}


def build_pipeline(config: Optional[Config] = None) -> Pipeline:
    """Build an sklearn Pipeline from Config object.

    Loads pipeline specification from Config.PREPROCESSING.PIPELINE_SPEC or 
    Config.PREPROCESSING_PIPELINE_SPEC and builds the corresponding sklearn Pipeline.
    
    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, will load from default location.
    
    Returns
    -------
    Pipeline
        sklearn Pipeline with configured preprocessing steps.
    
    Spec format example:
      {"steps": [ {"drop": {"columns": [...] }}, {"onehot": {...}}, {"scaler": {...}} ] }
    Supported step keys: drop, dropNullRows, onehot, scaler, dateSplit, index, imputer, sliding_window
    """
    # Load config if not provided
    if config is None:
        config = Config.load()
    
    # Extract pipeline spec from config
    spec = _get_pipeline_spec(config)
    
    steps: List = []

    for i, step in enumerate(spec.get("steps", [])):
        if not isinstance(step, dict) or len(step) == 0:
            continue
        key = next(iter(step.keys()))
        cfg = step[key] or {}

        if not isinstance(cfg, dict):
            continue

        if key == "dateSplit":
            cols = cfg.get("columns")
            if cols is None and isinstance(cfg.get("column"), str) and cfg.get("column").strip():
                cols = [cfg.get("column")]
            cols = cols or []
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            drop = cfg.get("dropColumns", True)
            steps.append((f"date_split_{i}", PipelineDateSpliter(columns=cols, dropColumns=drop)))
        elif key == "scaler":
            cols = cfg.get("columns", [])
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            steps.append((f"scaler_{i}", PipelineFeatureStandardScaler(columns=cols)))
        elif key == "imputer":
            cols = cfg.get("columns", [])
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            num_strategy = cfg.get("numeric_strategy", "mean")
            cat_strategy = cfg.get("categorical_strategy", "mode")
            fill_value = cfg.get("fill_value", None)
            steps.append((f"imputer_{i}", PipelineImputer(columns=cols, numeric_strategy=num_strategy, categorical_strategy=cat_strategy, fill_value=fill_value)))
        elif key in ("dropNullRows", "drop_null_rows"):
            cols = cfg.get("columns", [])
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            steps.append((f"drop_null_rows_{i}", PipelineNullRowDropper(columns=cols)))
        elif key == "onehot":
            cols = cfg.get("columns", [])
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            steps.append((f"onehot_{i}", PipelineOneHotEncoder(columns=cols)))
        elif key == "drop":
            cols = cfg.get("columns", [])
            # Filter out empty strings and check if any valid columns remain
            cols = [c for c in cols if isinstance(c, str) and c.strip()]
            if not isinstance(cols, (list, tuple)) or len(cols) == 0:
                continue
            steps.append((f"drop_{i}", PipelineFeatureDropper(columns=cols)))
        elif key in ("index", "set_index"):
            index_col = cfg.get("index") or cfg.get("column")
            if not isinstance(index_col, str) or index_col.strip() == "":
                continue
            steps.append((f"index_setter_{i}", PipelineIndexSetter(index=index_col)))
        elif key in ("sliding_window", "sequencer", "sequence"):
            column = cfg.get("column") or cfg.get("label")
            seq_len = cfg.get("sequence_length", cfg.get("sequenceLength", 60))
            sortlook = cfg.get("sortlook") or cfg.get("symbol_column")
            datetime_column = cfg.get("datetime_column") or cfg.get("datetimeColumn")
            if not isinstance(column, str) or column.strip() == "":
                continue
            if not isinstance(seq_len, int) or seq_len <= 0:
                continue
            if sortlook is not None and (not isinstance(sortlook, str) or not sortlook.strip()):
                sortlook = None
            if datetime_column is not None and (not isinstance(datetime_column, str) or not datetime_column.strip()):
                datetime_column = None
            steps.append((f"sliding_window_{i}", PipelineSlidingWindow(column=column, sequence_length=seq_len, sortlook=sortlook, datetime_column=datetime_column)))
        else:
            # unknown step key - ignore
            continue

    # If there are no valid steps, return an identity pipeline so callers
    # can safely call `fit_transform` without special-casing an empty spec.
    if len(steps) == 0:
        return Pipeline([("identity", FunctionTransformer(lambda X: X, validate=False))])

    return Pipeline(steps)

def save_pipeline(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)


def extract_sliding_window_step(pipeline: Pipeline):
    """Remove and return any PipelineSlidingWindow from a pipeline.

    Returns:
        (modified_pipeline, sliding_window_or_None)
    """
    remaining = []
    sliding_window = None
    for name, step in pipeline.steps:
        if isinstance(step, PipelineSlidingWindow):
            sliding_window = step
        else:
            remaining.append((name, step))

    if not remaining:
        remaining = [("identity", FunctionTransformer(lambda X: X, validate=False))]

    return Pipeline(remaining), sliding_window

# Deprecated alias
extract_sequencer_step = extract_sliding_window_step