"""
Base Model Training DAG Template Factory.

Provides create_model_dag() factory function that creates a model training
DAG with configurable experiment name and config path.

TODO: Full implementation - extract logic from 03_dag_model.py
"""
from datetime import datetime
import sys
from pathlib import Path
from typing import Optional

from airflow import DAG
from airflow.datasets import Dataset

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from config_load import Config


def _get_experiment_assets(experiment_name: Optional[str] = None):
    """Get experiment-scoped assets."""
    if experiment_name is None:
        from assets import TRANSFORMED_DATA_ASSET, TRAINED_MODEL_ASSET
        return TRANSFORMED_DATA_ASSET, TRAINED_MODEL_ASSET
    else:
        transformed_asset = Dataset(f"file://experiments/{experiment_name}/transformed_data.csv")
        trained_asset = Dataset(f"mlflow://experiments/{experiment_name}/model")
        return transformed_asset, trained_asset


def create_model_dag(
    experiment_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a model training DAG.

    Args:
        experiment_name: Optional experiment name for scoping.
        config_path: Optional path to experiment-specific config.json.

    Returns:
        Configured Airflow DAG object.

    Note: This is a stub. For full functionality, import and use the
    original 03_dag_model.py or fully implement this factory.
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    transformed_asset, trained_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    if experiment_name:
        dag_id = f"{experiment_name}_03_dag_model"
        description = f"Model training for {experiment_name} experiment"
        tags = ["model", "training", "ml-pipeline", "experiment", experiment_name]
    else:
        dag_id = "03_dag_model"
        description = "Model training DAG"
        tags = ["model", "training", "ml-pipeline"]

    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[transformed_asset] if experiment_name else None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    # Placeholder - actual implementation would include model training tasks
    # For generated experiments, the dashboard will create full DAG files

    return dag
