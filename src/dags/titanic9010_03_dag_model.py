"""
titanic9010_03_dag_model - Model Training DAG

Auto-generated experiment DAG for titanic9010.
Uses the template factory pattern from templates/base_03_model.py.
"""
import sys
from pathlib import Path

# Explicit airflow import for DAG discovery (required by Airflow's safe mode)
from airflow import DAG  # noqa: F401

# Add templates to path
dags_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dags_dir))

from templates.base_03_model import create_model_dag

# Create the DAG using the factory
dag = create_model_dag(
    experiment_name="titanic9010",
    config_path=str(dags_dir / "configs" / "titanic9010" / "config.json")
)
