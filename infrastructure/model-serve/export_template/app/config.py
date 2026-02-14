"""
Configuration loader for the standalone model serving app.

Reads config.json and provides typed access to data source,
data output, and model settings.
"""

import os
import json
import logging
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DatabaseSourceConfig(BaseModel):
    connection_string: str = ""
    query: str = ""


class FileSourceConfig(BaseModel):
    path: str = ""
    format: str = "csv"


class DataSourceConfig(BaseModel):
    type: str = "none"
    database: DatabaseSourceConfig = DatabaseSourceConfig()
    file: FileSourceConfig = FileSourceConfig()


class DatabaseOutputConfig(BaseModel):
    connection_string: str = ""
    table_name: str = "predictions_output"
    if_exists: str = "append"


class FileOutputConfig(BaseModel):
    path: str = ""
    format: str = "csv"


class DataOutputConfig(BaseModel):
    type: str = "response"
    database: DatabaseOutputConfig = DatabaseOutputConfig()
    file: FileOutputConfig = FileOutputConfig()


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


class ModelConfig(BaseModel):
    name: str = ""
    version: str = ""
    model_type: str = ""


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    server: ServerConfig = ServerConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    data_output: DataOutputConfig = DataOutputConfig()


def _strip_comments(obj):
    """Recursively strip keys starting with '_comment' from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(item) for item in obj]
    return obj


def load_config() -> AppConfig:
    """Load and parse config.json from CONFIG_PATH environment variable."""
    config_path = os.environ.get("CONFIG_PATH", "/app/config.json")
    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        raw = json.load(f)

    raw = _strip_comments(raw)
    config = AppConfig(**raw)

    logger.info(
        f"Config loaded: model={config.model.name}, "
        f"data_source={config.data_source.type}, "
        f"data_output={config.data_output.type}"
    )
    return config
