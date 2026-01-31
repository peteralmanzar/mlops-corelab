import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Config:
    """Loads configuration from src/config.json on each call.

    Validation rules:
    - Fail-fast for critical fields: MLFLOW.TRACKING_URI, DEFAULT_DAG_ARGS.retry_delay_seconds
    - Warn and default for non-critical fields.
    """

    def __init__(self, data: Dict[str, Any]):
        # assign all keys as attributes
        for k, v in data.items():
            setattr(self, k, v)

    @staticmethod
    def _config_path() -> Path:
        # src/utils/config_load.py -> src/config.json
        return Path(__file__).resolve().parents[1] / "config.json"

    @staticmethod
    def _model_config_path() -> Path:
        # src/utils/config_load.py -> src/config.model.json
        return Path(__file__).resolve().parents[1] / "config.model.json"

    @staticmethod
    def _normalize_path(p: Any) -> Any:
        if isinstance(p, str) and p:
            return os.path.abspath(os.path.expanduser(p))
        return p

    @classmethod
    def load(cls) -> "Config":
        # Load base configuration from config.model.json
        model_cfg_path = cls._model_config_path()
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model config file not found: {model_cfg_path}")

        with open(model_cfg_path, "r", encoding="utf-8") as f:
            merged = json.load(f)
        
        # Remove comment field if present
        merged.pop("_comment", None)

        # Overlay with user configuration from config.json (optional)
        cfg_path = cls._config_path()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Deep merge for nested dicts
            for k, v in data.items():
                if k in ("DEFAULT_DAG_ARGS", "DATA", "PREPROCESSING", "MODEL", "MLFLOW"):
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged.setdefault(k, {}).update(v)
                    else:
                        merged[k] = v
                else:
                    merged[k] = v
            logger.info("Loaded user configuration from %s", cfg_path)
        else:
            logger.warning("User config file not found at %s, using model defaults only", cfg_path)

        # Validation: critical fields
        # 1) MLFLOW.TRACKING_URI
        mlflow_cfg = merged.get("MLFLOW", {})
        if not mlflow_cfg.get("TRACKING_URI"):
            raise ValueError("Critical configuration missing: MLFLOW.TRACKING_URI")

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

        # Post-processing
        # Normalize paths in nested DATA, PREPROCESSING, and MODEL
        if "DATA" in merged and isinstance(merged["DATA"], dict):
            for path_key in ["RAW_PATH_FILE", "PROCESSED_PATH"]:
                if path_key in merged["DATA"]:
                    merged["DATA"][path_key] = cls._normalize_path(merged["DATA"][path_key])
        
        if "PREPROCESSING" in merged and isinstance(merged["PREPROCESSING"], dict):
            if "FEATURES_PATH" in merged["PREPROCESSING"]:
                merged["PREPROCESSING"]["FEATURES_PATH"] = cls._normalize_path(merged["PREPROCESSING"]["FEATURES_PATH"])
        
        if "MODEL" in merged and isinstance(merged["MODEL"], dict):
            if "ARTIFACTS_PATH" in merged["MODEL"]:
                merged["MODEL"]["ARTIFACTS_PATH"] = cls._normalize_path(merged["MODEL"]["ARTIFACTS_PATH"])

        # Ensure email is list
        email = merged.get("ALERT_EMAIL")
        if isinstance(email, str):
            merged["ALERT_EMAIL"] = [email]

        # Normalize LABEL_COLUMNS in MODEL: accept string or list, ensure list of strings
        if "MODEL" in merged and isinstance(merged["MODEL"], dict):
            lc = merged["MODEL"].get("LABEL_COLUMNS")
            if lc is None:
                merged["MODEL"]["LABEL_COLUMNS"] = ["target"]
            elif isinstance(lc, str):
                merged["MODEL"]["LABEL_COLUMNS"] = [lc]
            elif isinstance(lc, (list, tuple)):
                merged["MODEL"]["LABEL_COLUMNS"] = [str(x) for x in lc]
            else:
                raise ValueError("MODEL.LABEL_COLUMNS must be a string or a list/tuple of strings")

            # Validate elements
            if not isinstance(merged["MODEL"].get("LABEL_COLUMNS"), list) or not all(
                isinstance(x, str) and x.strip() for x in merged["MODEL"].get("LABEL_COLUMNS")
            ):
                raise ValueError("MODEL.LABEL_COLUMNS must be a non-empty list of non-empty strings")

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
        import copy
        log_copy = copy.deepcopy(merged)
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
