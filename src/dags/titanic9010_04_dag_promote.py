"""
titanic9010_04_dag_promote - Model Promotion DAG

Auto-generated experiment DAG for titanic9010.
Uses the template factory pattern from templates/base_04_promote.py.

Note: Model reload is handled by 00_dag_mlflow_watcher which monitors
MLflow Model Registry for champion alias changes across ALL experiments.
"""
import sys
from pathlib import Path

# Explicit airflow import for DAG discovery (required by Airflow's safe mode)
from airflow import DAG  # noqa: F401

# Add templates to path
dags_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dags_dir))

from templates.base_04_promote import create_promote_dag

# Create the DAG using the factory
dag = create_promote_dag(
    experiment_name="titanic9010",
    config_path=str(dags_dir / "configs" / "titanic9010" / "config.json")
)
