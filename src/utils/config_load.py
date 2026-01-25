import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


_DEFAULTS: Dict[str, Any] = {
    "config_version": 1,
    "MLFLOW":{
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_EXPERIMENT_NAME": "ml_pipeline",
    },
    "RANDOM_SEED": 42,
    "ALERT_EMAIL": ["admin@example.com"],
    "SLACK_WEBHOOK": None,
    "DEFAULT_DAG_ARGS": {
        "owner": "data-science-team",
        "depends_on_past": False,
        "email": ["admin@example.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay_seconds": 300,
    },
    "DATA": {
        "RAW_PATH_FILE": "/home/jovyan/data/raw",
        "PROCESSED_PATH": "/home/jovyan/data/processed",
        "FEATURES_PATH": "/home/jovyan/data/features",
        "TRAIN_TEST_SPLIT": 0.2,
        "VALIDATION_RULES": {
            "max_null_percentage": 0.1,
            "min_rows": 100,
            "required_columns": []
        },
        "FOLD_TYPE": None,
        "NUM_FOLDS": 5,
        "KFOLD_SHUFFLE": True,
        "KFOLD_RANDOM_STATE": 42,
        "TIME_COLUMN": None,
        "TIME_SERIES_GAP": 0,
        "TIME_SERIES_EXPANDING": True,
    },
    "PREPROCESSING":  {
        "PIPELINE_SPEC": {
            "steps": [
                {"dateSplit": {"columns": [], "dropColumns": True}},
                {"scaler": {"columns": []}},
                {"onehot": {"columns": []}},
                {"drop": {"columns": []}},
                {"index": {"column": ""}},
                {"sequencer": {"column": "", "sequence_length": 60}},
            ]
        },
    },
    "MODEL":{
        "ARTIFACTS_PATH": "/mlflow/artifacts",
        "LABEL_COLUMNS": ["target"],
        "PARAMS": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "VALIDATION_THRESHOLDS": {
            "min_accuracy": 0.75,
            "min_precision": 0.7,
            "min_recall": 0.7,
            "min_f1": 0.7,
            "max_inference_time_ms": 1000
        },
    },
}

class Config:
    """Loads configuration from src/dags/config.json on each call.

    Validation rules:
    - Fail-fast for critical fields: MLFLOW_TRACKING_URI, DEFAULT_DAG_ARGS.retry_delay_seconds,
      DAG_1_SCHEDULE and DAG_2_SCHEDULE.
    - Warn and default for non-critical fields.
    """

    def __init__(self, data: Dict[str, Any]):
        # assign all keys as attributes
        for k, v in data.items():
            setattr(self, k, v)

    @staticmethod
    def _config_path() -> Path:
        # src/dags/utils/config_loader.py -> src/dags/config.json
        return Path(__file__).resolve().parents[1] / "config.json"

    @staticmethod
    def _normalize_path(p: Any) -> Any:
        if isinstance(p, str) and p:
            return os.path.abspath(os.path.expanduser(p))
        return p

    @classmethod
    def load(cls) -> "Config":
        cfg_path = cls._config_path()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Merge defaults for missing non-critical fields (warn)
        merged = dict(_DEFAULTS)
        # shallow merge for DEFAULT_DAG_ARGS and other dicts
        for k, v in data.items():
            if k == "DEFAULT_DAG_ARGS":
                merged.setdefault("DEFAULT_DAG_ARGS", {}).update(v or {})
            else:
                merged[k] = v

        # Validation: critical fields
        # 1) MLFLOW_TRACKING_URI
        if not merged.get("MLFLOW_TRACKING_URI"):
            raise ValueError("Critical configuration missing: MLFLOW_TRACKING_URI")

        # 2) DEFAULT_DAG_ARGS.retry_delay_seconds
        dda = merged.get("DEFAULT_DAG_ARGS", {})
        rds = dda.get("retry_delay_seconds")
        if rds is None:
            raise ValueError("Critical configuration missing: DEFAULT_DAG_ARGS.retry_delay_seconds")
        try:
            rds_int = int(rds)
            if rds_int < 0:
                raise ValueError(
                    "DEFAULT_DAG_ARGS.retry_delay_seconds must be >= 0"
                )
        except Exception:
            raise ValueError(
                "DEFAULT_DAG_ARGS.retry_delay_seconds must be an integer number of seconds"
            )

        # 3) DAG schedules (DAG_1 and DAG_2 keys must exist;
        # None is allowed for manual/triggered DAGs)
        # Allow explicit null/None to indicate manual (no schedule),
        # but still require the keys to be present.
        if "DAG_1_SCHEDULE" not in merged:
            raise ValueError("Critical configuration missing: DAG_1_SCHEDULE")
        if "DAG_2_SCHEDULE" not in merged:
            raise ValueError("Critical configuration missing: DAG_2_SCHEDULE")

        # Post-processing
        # Normalize paths
        for path_key in [
            "DATA_RAW_PATH_FILE",
            "DATA_PROCESSED_PATH",
            "DATA_FEATURES_PATH",
            "MODEL_ARTIFACTS_PATH"
        ]:
            merged[path_key] = cls._normalize_path(merged.get(path_key))

        # Ensure email is list
        email = merged.get("ALERT_EMAIL")
        if isinstance(email, str):
            merged["ALERT_EMAIL"] = [email]

        # Normalize LABEL_COLUMNS: accept string or list, ensure list of strings
        lc = merged.get("LABEL_COLUMNS")
        if lc is None:
            merged["LABEL_COLUMNS"] = ["target"]
        elif isinstance(lc, str):
            merged["LABEL_COLUMNS"] = [lc]
        elif isinstance(lc, (list, tuple)):
            merged["LABEL_COLUMNS"] = [str(x) for x in lc]
        else:
            raise ValueError("LABEL_COLUMNS must be a string or a list/tuple of strings")

        # Validate elements
        if not isinstance(merged.get("LABEL_COLUMNS"), list) or not all(
            isinstance(x, str) and x.strip() for x in merged.get("LABEL_COLUMNS")
        ):
            raise ValueError("LABEL_COLUMNS must be a non-empty list of non-empty strings")

        # Convert retry_delay_seconds -> timedelta and attach to DEFAULT_DAG_ARGS.retry_delay
        merged.setdefault("DEFAULT_DAG_ARGS", {})
        merged["DEFAULT_DAG_ARGS"]["retry_delay"] = timedelta(seconds=int(merged["DEFAULT_DAG_ARGS"]["retry_delay_seconds"]))
        # Be resilient: if retry_delay was provided as a string (e.g. via overrides), coerce it.
        rd = merged["DEFAULT_DAG_ARGS"].get("retry_delay")
        if isinstance(rd, str):
            # Accept formats like "300", "300s", "5m", or "1h"
            import re

            s = rd.strip()
            m = re.match(r"^(?P<val>\d+)(?P<unit>[smh]?)$", s)
            if m:
                val = int(m.group("val"))
                unit = m.group("unit") or "s"
                mul = {"s": 1, "m": 60, "h": 3600}.get(unit, 1)
                merged["DEFAULT_DAG_ARGS"]["retry_delay"] = timedelta(seconds=val * mul)
            else:
                try:
                    merged["DEFAULT_DAG_ARGS"]["retry_delay"] = timedelta(seconds=int(s))
                except Exception:
                    raise ValueError(
                        "DEFAULT_DAG_ARGS.retry_delay must be a number of seconds "
                        "or a string like '5m', '300s', '1h'"
                    )
        # Keep the seconds key as well for logging/backwards compatibility

        # Log the loaded config (masking none-sensitive fields)
        log_copy = dict(merged)
        # Replace timedelta with seconds for logging
        try:
            rd = log_copy.get("DEFAULT_DAG_ARGS", {}).get("retry_delay")
            if isinstance(rd, timedelta):
                log_copy["DEFAULT_DAG_ARGS"]["retry_delay_seconds"] = int(rd.total_seconds())
                log_copy["DEFAULT_DAG_ARGS"]["retry_delay"] = f"{int(rd.total_seconds())}s"
        except Exception:
            pass

        logger.info(
            "Loaded configuration from %s:\n%s",
            cfg_path,
            json.dumps(log_copy, indent=2, default=str)
        )

        # Build Config object
        config_obj = cls(merged)
        return config_obj


__all__ = ["Config"]
