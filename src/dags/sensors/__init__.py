"""Custom Airflow sensors for ML pipeline."""
from .mlflow_champion_sensor import MlflowChampionSensor

__all__ = ['MlflowChampionSensor']
