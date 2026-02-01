"""
titanic9002_02_dag_preprocess - Preprocessing Pipeline DAG

Auto-generated experiment DAG for titanic9002.
Uses the template factory pattern from templates/base_preprocess.py.
"""
import sys
from pathlib import Path

# Explicit airflow import for DAG discovery (required by Airflow's safe mode)
from airflow import DAG  # noqa: F401

# Add templates to path
dags_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dags_dir))

from templates.base_preprocess import create_preprocess_dag

# Create the DAG using the factory
dag = create_preprocess_dag(
    experiment_name="titanic9002",
    config_path=str(dags_dir / "configs" / "titanic9002" / "config.json")
)
