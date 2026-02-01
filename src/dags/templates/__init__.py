"""
DAG Template Factories for ML Pipeline.

This module provides factory functions to create experiment-specific DAGs.
Each factory creates a parameterized DAG that can be customized per experiment.

Usage for existing DAGs (backward compatible):
    from templates.base_01_data import create_data_dag
    dag = create_data_dag()  # Uses default config

Usage for generated experiment DAGs:
    from templates.base_01_data import create_data_dag
    dag = create_data_dag(
        experiment_name="housing",
        config_path="/opt/airflow/dags/experiments/housing/config.json"
    )
"""
from .base_01_data import create_data_dag
from .base_02_preprocess import create_preprocess_dag
from .base_03_model import create_model_dag
from .base_04_promote import create_promote_dag

__all__ = [
    'create_data_dag',
    'create_preprocess_dag',
    'create_model_dag',
    'create_promote_dag',
]
