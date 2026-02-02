"""
Model Manager for FastAPI Serving

Discovers and loads champion models from MLflow Model Registry.
Supports sklearn pipelines, sklearn models, and Keras models.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import mlflow
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata."""
    name: str
    version: str
    run_id: str
    model: Any
    model_type: str  # 'sklearn_pipeline', 'sklearn_model', 'keras'
    loaded_at: datetime = field(default_factory=datetime.now)
    # Feature metadata fields
    input_features: Optional[List[str]] = None
    feature_dtypes: Optional[Dict[str, str]] = None
    feature_count: Optional[int] = None
    feature_metadata_available: bool = False

    def predict(self, data) -> Any:
        """Make prediction using the loaded model."""
        if self.model_type == 'keras':
            return self.model.predict(data)
        else:
            return self.model.predict(data)


class ModelManager:
    """
    Manages discovery and loading of champion models from MLflow.

    Discovers all registered models with a 'champion' alias and loads them
    for serving via the FastAPI endpoints.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        champion_alias: str = "champion"
    ):
        """
        Initialize the ModelManager.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to MLFLOW_TRACKING_URI env)
            champion_alias: Alias to look for (default: 'champion')
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        self.champion_alias = champion_alias
        self.models: Dict[str, LoadedModel] = {}
        self._client: Optional[MlflowClient] = None

        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"ModelManager initialized with tracking URI: {self.tracking_uri}")

    @property
    def client(self) -> MlflowClient:
        """Lazy-loaded MLflow client."""
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def _fetch_feature_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Fetch feature metadata from MLflow run artifacts and parameters.

        Looks for:
        - Parameters: preprocessed_train_columns, raw_data_columns, features
        - Artifacts: *_info.json, *_dtypes.json

        Args:
            run_id: MLflow run ID to fetch metadata from

        Returns:
            Dict with input_features, feature_dtypes, feature_count,
            and feature_metadata_available flag
        """
        try:
            run = self.client.get_run(run_id)
            params = run.data.params

            # Try to get feature columns from parameters (fastest approach)
            feature_columns = None
            for param_key in ['preprocessed_train_columns', 'raw_data_columns', 'features']:
                if param_key in params:
                    feature_columns = [col.strip() for col in params[param_key].split(',')]
                    logger.debug(f"Found feature columns in param '{param_key}'")
                    break

            # Try to fetch dtypes artifact
            feature_dtypes = None
            try:
                artifacts = self.client.list_artifacts(run_id)
                for artifact in artifacts:
                    if artifact.path.endswith('_dtypes.json'):
                        local_path = mlflow.artifacts.download_artifacts(
                            run_id=run_id,
                            artifact_path=artifact.path
                        )
                        with open(local_path, 'r') as f:
                            feature_dtypes = json.load(f)
                        logger.debug(f"Loaded dtypes from artifact '{artifact.path}'")
                        break
            except Exception as e:
                logger.debug(f"Could not fetch dtypes artifact: {e}")

            # If we couldn't find columns in params, try to get from info artifact
            if feature_columns is None:
                try:
                    artifacts = self.client.list_artifacts(run_id)
                    for artifact in artifacts:
                        if artifact.path.endswith('_info.json'):
                            local_path = mlflow.artifacts.download_artifacts(
                                run_id=run_id,
                                artifact_path=artifact.path
                            )
                            with open(local_path, 'r') as f:
                                info = json.load(f)
                            if 'columns' in info:
                                feature_columns = info['columns']
                                logger.debug(f"Found feature columns in artifact '{artifact.path}'")
                            break
                except Exception as e:
                    logger.debug(f"Could not fetch info artifact: {e}")

            if feature_columns:
                return {
                    'input_features': feature_columns,
                    'feature_dtypes': feature_dtypes,
                    'feature_count': len(feature_columns),
                    'feature_metadata_available': True
                }

            return {'feature_metadata_available': False}

        except Exception as e:
            logger.warning(f"Failed to fetch feature metadata for run {run_id}: {e}")
            return {'feature_metadata_available': False}

    def discover_champion_models(self) -> List[Dict[str, Any]]:
        """
        Discover all registered models that have a champion alias.

        Returns:
            List of dicts with model info (name, version, run_id)
        """
        champions = []

        try:
            # Get all registered models
            registered_models = self.client.search_registered_models()

            for rm in registered_models:
                model_name = rm.name
                try:
                    # Check if this model has a champion alias
                    mv = self.client.get_model_version_by_alias(
                        name=model_name,
                        alias=self.champion_alias
                    )
                    champions.append({
                        'name': model_name,
                        'version': mv.version,
                        'run_id': mv.run_id,
                        'aliases': list(mv.aliases) if hasattr(mv, 'aliases') and mv.aliases else [self.champion_alias]
                    })
                    logger.info(f"Found champion: {model_name} v{mv.version}")
                except Exception:
                    # Model doesn't have champion alias, skip it
                    logger.debug(f"No champion alias for model: {model_name}")
                    continue

        except Exception as e:
            logger.error(f"Error discovering champion models: {e}")
            raise

        return champions

    def load_model(self, model_name: str, version: str, run_id: str) -> LoadedModel:
        """
        Load a specific model version from MLflow.

        Tries loading in order: sklearn, keras, pyfunc.

        Args:
            model_name: Registered model name
            version: Model version
            run_id: MLflow run ID

        Returns:
            LoadedModel instance
        """
        model_uri = f"models:/{model_name}@{self.champion_alias}"
        model = None
        model_type = None

        # Try sklearn first (most common)
        try:
            logger.info(f"Trying to load {model_name} v{version} as sklearn...")
            model = mlflow.sklearn.load_model(model_uri)
            if isinstance(model, Pipeline):
                model_type = 'sklearn_pipeline'
            else:
                model_type = 'sklearn_model'
            logger.info(f"Successfully loaded as {model_type}")
        except Exception as e:
            logger.warning(f"sklearn load failed: {e}")

        # Try keras if sklearn failed
        if model is None:
            try:
                logger.info(f"Trying to load {model_name} v{version} as keras...")
                model = mlflow.keras.load_model(model_uri)
                model_type = 'keras'
                logger.info(f"Successfully loaded as keras")
            except Exception as e:
                logger.warning(f"keras load failed: {e}")

        # Try pyfunc as last resort
        if model is None:
            try:
                logger.info(f"Trying to load {model_name} v{version} as pyfunc...")
                model = mlflow.pyfunc.load_model(model_uri)
                model_type = 'pyfunc'
                logger.info(f"Successfully loaded as pyfunc")
            except Exception as e:
                logger.error(f"pyfunc load failed: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: could not load as sklearn, keras, or pyfunc")

        # Fetch feature metadata from MLflow
        feature_metadata = self._fetch_feature_metadata(run_id)
        logger.info(
            f"Feature metadata for {model_name}: "
            f"available={feature_metadata.get('feature_metadata_available', False)}, "
            f"count={feature_metadata.get('feature_count')}"
        )

        return LoadedModel(
            name=model_name,
            version=version,
            run_id=run_id,
            model=model,
            model_type=model_type,
            input_features=feature_metadata.get('input_features'),
            feature_dtypes=feature_metadata.get('feature_dtypes'),
            feature_count=feature_metadata.get('feature_count'),
            feature_metadata_available=feature_metadata.get('feature_metadata_available', False)
        )

    def load_all_champions(self) -> Dict[str, LoadedModel]:
        """
        Discover and load all champion models.

        Returns:
            Dictionary mapping model names to LoadedModel instances
        """
        start_time = time.time()
        logger.info("Loading all champion models...")

        # Discover champions
        champions = self.discover_champion_models()

        if not champions:
            logger.warning("No champion models found in MLflow registry")
            return {}

        # Load each champion
        loaded = {}
        for champ in champions:
            try:
                loaded_model = self.load_model(
                    model_name=champ['name'],
                    version=champ['version'],
                    run_id=champ['run_id']
                )
                loaded[champ['name']] = loaded_model
                logger.info(f"Loaded: {champ['name']} v{champ['version']}")
            except Exception as e:
                logger.error(f"Failed to load {champ['name']}: {e}")
                continue

        self.models = loaded
        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(loaded)} champion models in {elapsed:.2f}s")

        return loaded

    def reload_models(self) -> Dict[str, Any]:
        """
        Reload all champion models (for hot-reload functionality).

        Returns:
            Dictionary with reload status
        """
        old_count = len(self.models)
        old_versions = {name: m.version for name, m in self.models.items()}

        # Clear existing models
        self.models.clear()
        self._client = None  # Reset client

        # Reload
        self.load_all_champions()

        new_versions = {name: m.version for name, m in self.models.items()}

        # Determine what changed
        added = set(new_versions.keys()) - set(old_versions.keys())
        removed = set(old_versions.keys()) - set(new_versions.keys())
        updated = {
            name for name in set(old_versions.keys()) & set(new_versions.keys())
            if old_versions[name] != new_versions[name]
        }

        return {
            'status': 'success',
            'previous_count': old_count,
            'current_count': len(self.models),
            'added': list(added),
            'removed': list(removed),
            'updated': list(updated),
            'models': {name: m.version for name, m in self.models.items()}
        }

    def get_model(self, model_name: str) -> Optional[LoadedModel]:
        """
        Get a loaded model by name.

        Args:
            model_name: Model name

        Returns:
            LoadedModel or None if not found
        """
        return self.models.get(model_name)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all loaded models with their metadata.

        Returns:
            List of model info dictionaries
        """
        return [
            {
                'name': m.name,
                'version': m.version,
                'run_id': m.run_id,
                'model_type': m.model_type,
                'loaded_at': m.loaded_at.isoformat(),
                'feature_count': m.feature_count,
                'feature_metadata_available': m.feature_metadata_available
            }
            for m in self.models.values()
        ]

    def get_model_features(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed feature information for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict with feature details or None if model not found
        """
        model = self.models.get(model_name)
        if model is None:
            return None

        return {
            'model_name': model.name,
            'model_version': model.version,
            'feature_metadata_available': model.feature_metadata_available,
            'feature_count': model.feature_count,
            'input_features': model.input_features,
            'feature_dtypes': model.feature_dtypes
        }

    def is_ready(self) -> bool:
        """Check if any models are loaded and ready to serve."""
        return len(self.models) > 0
