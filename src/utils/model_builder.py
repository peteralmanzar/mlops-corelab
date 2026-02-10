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
from data_transform import PipelineFeatureDropper, PipelineSequencer
from typing import Optional, Dict, Any
from model_template import (
    GetModelTemplateMLPRegression,
    GetModelTemplateMLPBinaryClassification,
    GetModelTemplateMLPMultiClassification,
    GetModelTemplateLSTM,
    GetModelTemplateLSTMBinaryClassification,
    GetModelTemplateLSTMMultiClassification,
    build_dynamic_mlp,
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
            tracking_uri=self.config.MLFLOW.get("TRACKING_URI"),
            experiment_name=self.config.MLFLOW.get("EXPERIMENT_NAME")
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

    def build_model_for_training(
        self,
        task_type: str,
        num_features: int,
        num_classes: int,
        hyperparams: Optional[Dict[str, Any]] = None,
        sequence_length: Optional[int] = None
    ):
        """
        Instantiate a fresh Keras model based on task type and configuration.

        Args:
            task_type: 'regression', 'binary_classification', or 'multi_classification'
            num_features: Number of input features
            num_classes: Number of output classes
            hyperparams: Optional dict with 'architecture' and 'training' keys from Optuna tuning.
                         If provided, uses dynamic model building with tuned hyperparameters.
            sequence_length: If set, builds an LSTM model for sequential/3D input data.

        Returns:
            Compiled Keras model
        """
        optimizer_str = self.config.MODEL.get("OPTIMIZER", "adam")
        optimizer = self._resolve_optimizer(optimizer_str)

        # LSTM models for sequenced (3D) data
        if sequence_length is not None:
            # If hyperparams provided (from Optuna), build a dynamic LSTM
            if hyperparams is not None:
                from model_template import build_dynamic_lstm
                arch_params = hyperparams.get('architecture', {})
                train_params = hyperparams.get('training', {})
                hp_optimizer_str = train_params.get('optimizer', optimizer_str)
                hp_optimizer = self._resolve_optimizer(hp_optimizer_str)
                learning_rate = train_params.get('learning_rate', 0.001)
                return build_dynamic_lstm(
                    sequence_length=sequence_length,
                    num_features=num_features,
                    task_type=task_type,
                    num_classes=num_classes,
                    architecture_params=arch_params,
                    optimizer=hp_optimizer,
                    learning_rate=learning_rate
                )
            # Fallback to hardcoded LSTM templates
            if task_type == 'regression':
                return GetModelTemplateLSTM(
                    numberOfSteps=sequence_length, numberOfFeatures=num_features, optimizer=optimizer)
            elif task_type == 'binary_classification':
                return GetModelTemplateLSTMBinaryClassification(
                    numberOfSteps=sequence_length, numberOfFeatures=num_features, optimizer=optimizer)
            elif task_type == 'multi_classification':
                return GetModelTemplateLSTMMultiClassification(
                    numberOfSteps=sequence_length, numberOfFeatures=num_features,
                    num_classes=num_classes, optimizer=optimizer)
            else:
                raise ValueError(f"Unknown task_type for LSTM model build: {task_type}")

        # If hyperparams provided (from Optuna), use dynamic model builder
        if hyperparams is not None:
            arch_params = hyperparams.get('architecture', {})
            train_params = hyperparams.get('training', {})

            # Override optimizer and learning rate from hyperparams
            optimizer_str = train_params.get('optimizer', optimizer_str)
            optimizer = self._resolve_optimizer(optimizer_str)
            learning_rate = train_params.get('learning_rate', 0.001)

            return build_dynamic_mlp(
                num_features=num_features,
                task_type=task_type,
                num_classes=num_classes,
                architecture_params=arch_params,
                optimizer=optimizer,
                learning_rate=learning_rate
            )

        # Fallback to original hardcoded templates
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

    def _get_sequencer_helper_columns(self):
        """Get columns used by the sequencer (sortlook, datetime_column) that
        should be dropped before the model in a combined inference pipeline."""
        cols = []
        spec = getattr(self.config, 'PREPROCESSING', {})
        if isinstance(spec, dict):
            spec = spec.get('PIPELINE_SPEC', {})
        else:
            spec = {}
        for step in spec.get('steps', []):
            if not isinstance(step, dict):
                continue
            key = next(iter(step.keys()), None)
            if key in ('sequencer', 'sequence'):
                cfg = step[key] or {}
                for field in ('sortlook', 'symbol_column', 'datetime_column', 'datetimeColumn'):
                    val = cfg.get(field)
                    if val and isinstance(val, str) and val.strip():
                        cols.append(val)
        return cols

    def combine_pipeline_and_model(self, pipeline: Pipeline, keras_model, save_to_disk: bool = True) -> Pipeline:
        """
        Combine preprocessing pipeline and a trained Keras model into an sklearn Pipeline.
        Saves combined pipeline to artifacts path and returns the combined pipeline object.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be an sklearn.pipeline.Pipeline instance")

        wrapper = KerasModelWrapper(keras_model)

        # Check if preprocessing pipeline already contains a sequencer.
        # When present, the sequencer's transform() handles dropping helper
        # columns (sortlook, datetime) and producing 3D input for the model.
        has_sequencer = any(
            isinstance(step, PipelineSequencer) for _, step in pipeline.steps
        )

        steps = [('preprocessing', pipeline)]
        if not has_sequencer:
            # Non-sequence models: drop helper columns if any remain
            sequencer_cols = self._get_sequencer_helper_columns()
            if sequencer_cols:
                steps.append(('drop_sequencer_cols', PipelineFeatureDropper(columns=sequencer_cols)))
        steps.append(('model', wrapper))
        combined = Pipeline(steps)

        artifacts_path = self.config.MODEL.get("ARTIFACTS_PATH", os.path.join(os.getcwd(), "artifacts"))
        os.makedirs(artifacts_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_path = os.path.join(artifacts_path, f"combined_pipeline_{timestamp}.joblib")

        if save_to_disk:
            joblib.dump(combined, combined_path)

        return combined, combined_path


__all__ = ["ModelBuilder", "KerasModelWrapper"]
