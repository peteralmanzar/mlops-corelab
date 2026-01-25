"""Preprocessing pipeline factory and persistence helpers.

Builds sklearn-style pipelines from a simple spec and provides save/load helpers.
"""
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
from typing import Dict, Any, List

from utils.data_transform import (
        PipelineOneHotEncoder,
        PipelineFeatureStandardScaler,
        PipelineFeatureDropper,
        PipelineDateSpliter,
        PipelineIndexSetter,
        PipelineSequencer,
        PipelineImputer,
    )

def build_pipeline(spec: Dict[str, Any]) -> Pipeline:
    """Build an sklearn Pipeline from a spec dict.

    Spec format example:
      {"steps": [ {"drop": {"columns": [...] }}, {"onehot": {...}}, {"scaler": {...}} ] }
    Supported step keys: drop, onehot, scaler, dateSplit, index, dataSplit, datasplit_to_numpy, sequencer
    """
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
        elif key in ("sequencer", "sequence"):
            column = cfg.get("column") or cfg.get("label")
            seq_len = cfg.get("sequence_length", cfg.get("sequenceLength", 60))
            if not isinstance(column, str) or column.strip() == "":
                continue
            if not isinstance(seq_len, int) or seq_len <= 0:
                continue
            steps.append((f"sequencer_{i}", PipelineSequencer(column=column, sequence_length=seq_len)))
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