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


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    transformed_asset = Dataset(f"file://experiments/{experiment_name}/transformed_data.csv")
    trained_asset = Dataset(f"mlflow://experiments/{experiment_name}/model")
    return transformed_asset, trained_asset


def create_model_dag(
    experiment_name: str,
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
    dag_id = f"{experiment_name}_03_dag_model"
    description = f"Model training for {experiment_name} experiment"
    tags = ["model", "training", "ml-pipeline", "experiment", experiment_name]

    from airflow.operators.python import PythonOperator

    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[transformed_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    with dag:
        # Placeholder task - actual implementation would include model training tasks
        # For generated experiments, the dashboard will create full DAG files
        def _placeholder_train(**kwargs):
            print(f"Model training placeholder for {experiment_name or 'default'}")

        train_task = PythonOperator(
            task_id="train_model",
            python_callable=_placeholder_train,
            outlets=[trained_asset],
        )

    return dag
