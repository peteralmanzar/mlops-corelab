import os
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from config_load import Config
from mlflow_log import MLFlowLogger
from model_template import (
    GetModelTemplateMLPRegression,
    GetModelTemplateMLPBinaryClassification,
    GetModelTemplateMLPMultiClassification,
    Optimizer as TemplateOptimizer,
)


class KerasModelWrapper(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for a trained Keras model.
    Used so the Keras model can be the final step in an sklearn Pipeline.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure numpy array
        arr = X.values if hasattr(X, 'values') else np.asarray(X)
        preds = self.model.predict(arr)
        # If preds is 2D single-column, return as 2D array
        return preds


class ModelBuilder:
    """
    Utility to build Keras models for training and to combine a trained
    Keras model with a preprocessing sklearn Pipeline into a single
    sklearn Pipeline object that can be logged and registered.
    """
    def __init__(self, config: Config = None):
        self.config = config or Config.load()
        self.mlflow_logger = MLFlowLogger(
            tracking_uri=self.config.MLFLOW.get("MLFLOW_TRACKING_URI"),
            experiment_name=self.config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
        )

    def _resolve_optimizer(self, optimizer_str: str):
        if not optimizer_str:
            return TemplateOptimizer.ADAM
        opt = optimizer_str.lower()
        mapping = {
            "adam": TemplateOptimizer.ADAM,
            "sgd": TemplateOptimizer.SGD,
            "rmsprop": TemplateOptimizer.RMS_PROP,
            "adadelta": TemplateOptimizer.ADA_DELTA,
            "adagrad": TemplateOptimizer.ADA_GRAD,
            "adamax": TemplateOptimizer.ADA_MAX,
            "nadam": TemplateOptimizer.NADAM,
            "ftrl": TemplateOptimizer.FTRL,
        }
        return mapping.get(opt, TemplateOptimizer.ADAM)

    def build_model_for_training(self, task_type: str, num_features: int, num_classes: int):
        """
        Instantiate a fresh Keras model based on task type and configuration.
        """
        optimizer_str = self.config.MODEL.get("OPTIMIZER", "adam")
        optimizer = self._resolve_optimizer(optimizer_str)

        if task_type == 'regression':
            model = GetModelTemplateMLPRegression(numberOfFeatures=num_features, optimizer=optimizer)
        elif task_type == 'binary_classification':
            model = GetModelTemplateMLPBinaryClassification(numberOfFeatures=num_features, optimizer=optimizer)
        elif task_type == 'multi_classification':
            model = GetModelTemplateMLPMultiClassification(numberOfFeatures=num_features, num_classes=num_classes, optimizer=optimizer)
        else:
            raise ValueError(f"Unknown task_type for model build: {task_type}")

        return model

    def load_preprocessing_pipeline(self, pipeline_run_id: str, artifact_path: str = "preprocessing_pipeline") -> Pipeline:
        """
        Load sklearn preprocessing pipeline from MLflow run artifacts.
        """
        if not pipeline_run_id:
            raise ValueError("pipeline_run_id is required to load preprocessing pipeline")

        model_uri = f"runs:/{pipeline_run_id}/{artifact_path}"

        # validate artifact exists via MLFlowLogger helper
        if not self.mlflow_logger.validate_artifact_exists(model_uri):
            raise FileNotFoundError(f"Preprocessing pipeline artifact not found at {model_uri}")

        pipeline = MLFlowLogger.load_sklearn_pipeline(model_uri)
        return pipeline

    def load_trained_model(self, model_run_id: str, artifact_path: str = "model"):
        """
        Load a trained Keras model from MLflow run artifacts.
        """
        if not model_run_id:
            raise ValueError("model_run_id is required to load trained model")

        model_uri = f"runs:/{model_run_id}/{artifact_path}"

        if not self.mlflow_logger.validate_artifact_exists(model_uri):
            raise FileNotFoundError(f"Trained model artifact not found at {model_uri}")

        model = MLFlowLogger.load_keras_model(model_uri)
        return model

    def combine_pipeline_and_model(self, pipeline: Pipeline, keras_model, save_to_disk: bool = True) -> Pipeline:
        """
        Combine preprocessing pipeline and a trained Keras model into an sklearn Pipeline.
        Saves combined pipeline to artifacts path and returns the combined pipeline object.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be an sklearn.pipeline.Pipeline instance")

        wrapper = KerasModelWrapper(keras_model)
        combined = Pipeline([('preprocessing', pipeline), ('model', wrapper)])

        artifacts_path = self.config.MODEL.get("ARTIFACTS_PATH", os.path.join(os.getcwd(), "artifacts"))
        os.makedirs(artifacts_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_path = os.path.join(artifacts_path, f"combined_pipeline_{timestamp}.joblib")

        if save_to_disk:
            joblib.dump(combined, combined_path)

        return combined, combined_path


__all__ = ["ModelBuilder", "KerasModelWrapper"]
