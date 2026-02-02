"""
Base Model Promotion DAG Template Factory.

Provides create_promote_dag() factory function that creates a model promotion
DAG with configurable experiment name and config path.

TODO: Full implementation - extract logic from 04_dag_promote.py
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


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    trained_asset = Dataset(f"mlflow://experiments/{experiment_name}/model")
    return trained_asset


def create_promote_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a model promotion DAG.

    Args:
        experiment_name: Optional experiment name for scoping.
        config_path: Optional path to experiment-specific config.json.

    Returns:
        Configured Airflow DAG object.

    Note: This is a stub. For full functionality, import and use the
    original 04_dag_promote.py or fully implement this factory.

    Important: This DAG does NOT emit PROMOTED_MODEL_ASSET.
    Model reload is handled by 00_dag_mlflow_watcher which monitors
    MLflow Model Registry for champion alias changes.
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    trained_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_04_dag_promote"
    description = f"Model promotion for {experiment_name} experiment"
    tags = ["promotion", "ml-pipeline", "experiment", experiment_name]

    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[trained_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    # Placeholder - actual implementation would include promotion tasks
    # For generated experiments, the dashboard will create full DAG files
    #
    # Note: Champion model reload is handled by 00_dag_mlflow_watcher,
    # which monitors MLflow for champion alias changes across ALL experiments.

    return dag
