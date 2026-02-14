"""
Data source readers for batch predictions.

Supports reading from databases (via SQLAlchemy) and files (CSV/JSON).
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

from .config import DataSourceConfig

logger = logging.getLogger(__name__)


def read_batch_data(config: DataSourceConfig) -> pd.DataFrame:
    """
    Read data from the configured source.

    Args:
        config: Data source configuration

    Returns:
        DataFrame with input data

    Raises:
        ValueError: If source type is 'none' or not configured
    """
    source_type = config.type.lower().strip()

    if source_type == "none":
        raise ValueError(
            "No data source configured. Set data_source.type in config.json "
            "to 'database', 'csv', or 'json', or provide a source_override in the request."
        )

    if source_type == "database":
        return _read_from_database(config)
    elif source_type == "csv":
        return _read_from_csv(config)
    elif source_type == "json":
        return _read_from_json(config)
    else:
        raise ValueError(f"Unsupported data source type: '{source_type}'. Use 'database', 'csv', or 'json'.")


def _read_from_database(config: DataSourceConfig) -> pd.DataFrame:
    if not config.database.connection_string:
        raise ValueError("Database connection_string is empty in config")
    if not config.database.query:
        raise ValueError("Database query is empty in config")

    logger.info(f"Reading from database: {config.database.query[:80]}...")
    engine = create_engine(config.database.connection_string)
    df = pd.read_sql(config.database.query, engine)
    engine.dispose()
    logger.info(f"Read {len(df)} rows from database")
    return df


def _read_from_csv(config: DataSourceConfig) -> pd.DataFrame:
    if not config.file.path:
        raise ValueError("File path is empty in config")

    logger.info(f"Reading CSV from: {config.file.path}")
    df = pd.read_csv(config.file.path)
    logger.info(f"Read {len(df)} rows from CSV")
    return df


def _read_from_json(config: DataSourceConfig) -> pd.DataFrame:
    if not config.file.path:
        raise ValueError("File path is empty in config")

    logger.info(f"Reading JSON from: {config.file.path}")
    df = pd.read_json(config.file.path)
    logger.info(f"Read {len(df)} rows from JSON")
    return df
