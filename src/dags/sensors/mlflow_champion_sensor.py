"""
MLflow Champion Sensor - Monitors MLflow Model Registry for champion alias changes.

This sensor polls the MLflow Model Registry to detect when any registered model
has a new champion version. It enables decoupled model serving updates that work
across all experiments.
"""
import json
from typing import Any, Dict, Optional

from airflow.models import Variable
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.context import Context

import mlflow
from mlflow.tracking import MlflowClient


class MlflowChampionSensor(BaseSensorOperator):
    """
    Sensor that monitors MLflow Model Registry for champion alias changes.

    This sensor:
    1. Queries all registered models in MLflow
    2. Checks for models with a 'champion' alias
    3. Compares current champion versions against last known state
    4. Returns True (poke succeeds) when any champion version changes

    The last known state is stored in an Airflow Variable for persistence
    across DAG runs.

    Args:
        tracking_uri: MLflow tracking server URI
        champion_alias: Alias to monitor (default: 'champion')
        state_variable_key: Airflow Variable key for storing state
    """

    template_fields = ('tracking_uri',)

    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        champion_alias: str = "champion",
        state_variable_key: str = "mlflow_champion_state",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.champion_alias = champion_alias
        self.state_variable_key = state_variable_key

    def _get_current_champions(self) -> Dict[str, Dict[str, Any]]:
        """
        Query MLflow for all models with champion aliases.

        Returns:
            Dict mapping model names to their champion version info:
            {
                "model_name": {
                    "version": "3",
                    "run_id": "abc123",
                    "creation_timestamp": 1234567890
                }
            }
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        champions = {}

        try:
            # Get all registered models
            registered_models = client.search_registered_models()

            for rm in registered_models:
                model_name = rm.name

                try:
                    # Try to get the version with champion alias
                    version = client.get_model_version_by_alias(
                        name=model_name,
                        alias=self.champion_alias
                    )

                    champions[model_name] = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "creation_timestamp": version.creation_timestamp
                    }

                except mlflow.exceptions.MlflowException:
                    # Model doesn't have a champion alias - skip it
                    pass

        except Exception as e:
            self.log.warning(f"Error querying MLflow: {e}")

        return champions

    def _get_last_known_state(self) -> Dict[str, Dict[str, Any]]:
        """Load last known champion state from Airflow Variable."""
        try:
            state_json = Variable.get(self.state_variable_key, default_var="{}")
            return json.loads(state_json)
        except Exception as e:
            self.log.warning(f"Could not load state variable: {e}")
            return {}

    def _save_current_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Save current champion state to Airflow Variable."""
        try:
            Variable.set(self.state_variable_key, json.dumps(state))
        except Exception as e:
            self.log.error(f"Could not save state variable: {e}")

    def _detect_changes(
        self,
        current: Dict[str, Dict[str, Any]],
        previous: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare current and previous states to detect changes.

        Returns:
            Dict with change information:
            {
                "has_changes": bool,
                "new_champions": ["model1", "model2"],
                "updated_champions": ["model3"],
                "removed_champions": ["model4"]
            }
        """
        current_models = set(current.keys())
        previous_models = set(previous.keys())

        new_champions = current_models - previous_models
        removed_champions = previous_models - current_models

        # Check for version updates in existing models
        updated_champions = []
        for model_name in current_models & previous_models:
            if current[model_name]["version"] != previous[model_name]["version"]:
                updated_champions.append(model_name)

        has_changes = bool(new_champions or updated_champions or removed_champions)

        return {
            "has_changes": has_changes,
            "new_champions": list(new_champions),
            "updated_champions": updated_champions,
            "removed_champions": list(removed_champions)
        }

    def poke(self, context: Context) -> bool:
        """
        Check if any champion versions have changed.

        Returns True if changes detected, False otherwise.
        """
        self.log.info(f"Checking MLflow for champion changes at {self.tracking_uri}")

        # Get current state from MLflow
        current_state = self._get_current_champions()
        self.log.info(f"Current champions: {list(current_state.keys())}")

        # Get last known state
        previous_state = self._get_last_known_state()
        self.log.info(f"Previous champions: {list(previous_state.keys())}")

        # Detect changes
        changes = self._detect_changes(current_state, previous_state)

        if changes["has_changes"]:
            self.log.info("Champion changes detected!")
            if changes["new_champions"]:
                self.log.info(f"  New champions: {changes['new_champions']}")
            if changes["updated_champions"]:
                self.log.info(f"  Updated champions: {changes['updated_champions']}")
            if changes["removed_champions"]:
                self.log.info(f"  Removed champions: {changes['removed_champions']}")

            # Save current state for next comparison
            self._save_current_state(current_state)

            # Push change details to XCom for downstream tasks
            context['ti'].xcom_push(key='champion_changes', value=changes)
            context['ti'].xcom_push(key='current_champions', value=current_state)

            return True
        else:
            self.log.info("No champion changes detected")
            return False
