"""
Data sink writers for batch prediction output.

Supports writing to databases (via SQLAlchemy) and files (CSV/JSON).
Output type can be compound (e.g. "database+response") to write to
multiple destinations.
"""

import logging
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine

from .config import DataOutputConfig

logger = logging.getLogger(__name__)


def write_predictions(df: pd.DataFrame, config: DataOutputConfig) -> Dict[str, List[str]]:
    """
    Write prediction results to configured destinations.

    Args:
        df: DataFrame with input data + prediction column
        config: Output configuration

    Returns:
        Dict with 'destinations' key listing where data was written
    """
    output_types = [t.strip().lower() for t in config.type.split("+")]
    destinations = []

    if "database" in output_types:
        _write_to_database(df, config)
        destinations.append("database")

    if "csv" in output_types:
        _write_to_csv(df, config)
        destinations.append("csv")

    if "json" in output_types:
        _write_to_json(df, config)
        destinations.append("json")

    return {"destinations": destinations}


def _write_to_database(df: pd.DataFrame, config: DataOutputConfig) -> None:
    if not config.database.connection_string:
        raise ValueError("Output database connection_string is empty in config")
    if not config.database.table_name:
        raise ValueError("Output database table_name is empty in config")

    logger.info(
        f"Writing {len(df)} rows to database table: {config.database.table_name} "
        f"(if_exists={config.database.if_exists})"
    )
    engine = create_engine(config.database.connection_string)
    df.to_sql(
        config.database.table_name,
        engine,
        if_exists=config.database.if_exists,
        index=False
    )
    engine.dispose()
    logger.info("Database write complete")


def _write_to_csv(df: pd.DataFrame, config: DataOutputConfig) -> None:
    if not config.file.path:
        raise ValueError("Output file path is empty in config")

    logger.info(f"Writing {len(df)} rows to CSV: {config.file.path}")
    df.to_csv(config.file.path, index=False)
    logger.info("CSV write complete")


def _write_to_json(df: pd.DataFrame, config: DataOutputConfig) -> None:
    if not config.file.path:
        raise ValueError("Output file path is empty in config")

    logger.info(f"Writing {len(df)} rows to JSON: {config.file.path}")
    df.to_json(config.file.path, orient="records", indent=2)
    logger.info("JSON write complete")
