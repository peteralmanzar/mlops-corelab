"""
Base Preprocessing DAG Template Factory.

Provides create_preprocess_dag() factory function that creates a preprocessing
pipeline DAG with configurable experiment name and config path.

TODO: Full implementation - extract logic from 02_dag_preprocess.py
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
    split_asset = Dataset(f"file://experiments/{experiment_name}/train_test_split.csv")
    pipeline_asset = Dataset(f"file://experiments/{experiment_name}/preprocessing_pipeline.joblib")
    transformed_asset = Dataset(f"file://experiments/{experiment_name}/transformed_data.csv")
    return split_asset, pipeline_asset, transformed_asset


def create_preprocess_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a preprocessing pipeline DAG.

    Args:
        experiment_name: Optional experiment name for scoping.
        config_path: Optional path to experiment-specific config.json.

    Returns:
        Configured Airflow DAG object.

    Note: This is a stub. For full functionality, import and use the
    original 02_dag_preprocess.py or fully implement this factory.
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    split_asset, pipeline_asset, transformed_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_02_dag_preprocess"
    description = f"Preprocessing pipeline for {experiment_name} experiment"
    tags = ["preprocessing", "ml-pipeline", "experiment", experiment_name]

    # Import the actual implementation
    # For now, we delegate to the original DAG logic
    # TODO: Extract task functions like base_data.py

    from airflow.operators.python import PythonOperator

    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[split_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    with dag:
        # Placeholder task - actual implementation would include all preprocessing tasks
        # For generated experiments, the dashboard will create full DAG files
        def _placeholder_preprocess(**kwargs):
            print(f"Preprocessing placeholder for {experiment_name or 'default'}")

        preprocess_task = PythonOperator(
            task_id="preprocess_data",
            python_callable=_placeholder_preprocess,
            outlets=[transformed_asset],
        )

    return dag
