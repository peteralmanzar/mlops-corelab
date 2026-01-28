"""
MLflow Logger Utility

Provides a wrapper class for MLflow tracking operations including
experiment setup, run management, and logging of metrics, parameters,
and artifacts for data exploration and model training tasks.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class MLFlowLogger:
    """
    MLflow logging utility for tracking experiments, metrics, and artifacts.
    
    Handles MLflow run lifecycle, logging of dataset information, metrics,
    parameters, and artifacts with automatic experiment setup.
    """
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLFlowLogger with tracking URI and experiment name.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_id = None
        self.run_name = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Error setting up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to log with the run
            
        Returns:
            Run ID of the started run
        """
        self.run_name = run_name
        mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = mlflow.active_run().info.run_id
        print(f"Started MLflow run: {self.run_id} (Name: {run_name})")
        return self.run_id
    
    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Log comprehensive dataset information similar to dataset.info().
        
        Logs:
        - Column names
        - Column data types
        - Total number of rows
        - Number of non-null rows per column
        - Number of null rows per column
        - Memory usage
        
        Args:
            df: Pandas DataFrame to analyze
            dataset_name: Name prefix for logged metrics
            
        Returns:
            Dictionary containing all dataset information
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Basic shape information
        num_rows, num_cols = df.shape
        mlflow.log_param(f"{dataset_name}_num_rows", num_rows)
        mlflow.log_param(f"{dataset_name}_num_columns", num_cols)
        
        # Column information
        column_info = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = int(df[col].count())
            null_count = int(df[col].isna().sum())
            
            column_info[col] = {
                "dtype": dtype,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": round((null_count / num_rows) * 100, 2)
            }
            
            # Log individual column metrics
            mlflow.log_metric(f"{dataset_name}_col_{col}_non_null_count", non_null_count)
            mlflow.log_metric(f"{dataset_name}_col_{col}_null_count", null_count)
            mlflow.log_metric(f"{dataset_name}_col_{col}_null_pct", column_info[col]["null_percentage"])
        
        # Create summary info dictionary
        dataset_info = {
            "dataset_name": dataset_name,
            "num_rows": num_rows,
            "num_columns": num_cols,
            "columns": list(df.columns),
            "column_info": column_info,
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
        }
        
        # Log column names and dtypes as parameters
        mlflow.log_param(f"{dataset_name}_columns", ",".join(df.columns))
        dtypes_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}
        mlflow.log_dict(dtypes_summary, f"{dataset_name}_dtypes.json")
        
        # Log full dataset info as JSON artifact
        mlflow.log_dict(dataset_info, f"{dataset_name}_info.json")
        
        print(f"Logged dataset info for '{dataset_name}': {num_rows} rows, {num_cols} columns")
        return dataset_info
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """
        Log multiple metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for metric history
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)
        
        print(f"Logged {len(metrics)} metrics to MLflow")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        for param_name, param_value in params.items():
            # Convert to string if necessary
            if isinstance(param_value, (list, dict)):
                param_value = json.dumps(param_value)
            mlflow.log_param(param_name, param_value)
        
        print(f"Logged {len(params)} parameters to MLflow")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a single artifact file to MLflow.
        
        Args:
            local_path: Local file path to log
            artifact_path: Optional subdirectory in artifact store
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        print(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all artifacts in a directory to MLflow.
        
        Args:
            local_dir: Local directory path containing artifacts
            artifact_path: Optional subdirectory in artifact store
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        print(f"Logged artifacts from directory: {local_dir}")
    
    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Log a pandas DataFrame as a CSV artifact.
        
        Args:
            df: DataFrame to log
            filename: Name for the CSV file (should end in .csv)
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Create temporary file
        temp_path = Path("/tmp") / filename
        df.to_csv(temp_path, index=False)
        
        # Log as artifact
        mlflow.log_artifact(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        print(f"Logged DataFrame as artifact: {filename}")
    
    def log_text(self, text: str, artifact_file: str):
        """
        Log text content as an artifact.
        
        Args:
            text: Text content to log
            artifact_file: Name of the artifact file
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.log_text(text, artifact_file)
        print(f"Logged text artifact: {artifact_file}")
    
    def log_sklearn_pipeline(self, pipeline: Pipeline, artifact_path: str = "pipeline", 
                             registered_model_name: Optional[str] = None,
                             signature=None, input_example: Optional[pd.DataFrame] = None) -> None:
        """
        Log a scikit-learn Pipeline to MLflow with type safety.
        
        Args:
            pipeline: Scikit-learn Pipeline object
            artifact_path: Path within the artifact URI to save the pipeline
            registered_model_name: Optional name for model registration
            signature: Optional MLflow model signature
            input_example: Optional pandas DataFrame showing example input
        
        Raises:
            RuntimeError: If no active MLflow run
            TypeError: If pipeline is not a sklearn Pipeline instance
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Expected sklearn.pipeline.Pipeline, got {type(pipeline).__name__}")
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        print(f"Logged sklearn Pipeline to artifact path: {artifact_path}")
        print(f"  Pipeline steps: {[step[0] for step in pipeline.steps]}")
        if registered_model_name:
            print(f"  Registered as: {registered_model_name}")
    
    def log_sklearn_model(self, model: BaseEstimator, artifact_path: str = "model", 
                          registered_model_name: Optional[str] = None,
                          signature=None, input_example: Optional[pd.DataFrame] = None) -> None:
        """
        Log a scikit-learn model (classifier/regressor) to MLflow with type safety.
        
        Args:
            model: Scikit-learn estimator/model object (e.g., RandomForest, LogisticRegression)
            artifact_path: Path within the artifact URI to save the model
            registered_model_name: Optional name for model registration
            signature: Optional MLflow model signature
            input_example: Optional pandas DataFrame showing example input
        
        Raises:
            RuntimeError: If no active MLflow run
            TypeError: If model is not a sklearn estimator
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        if not isinstance(model, BaseEstimator):
            raise TypeError(f"Expected sklearn BaseEstimator, got {type(model).__name__}")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        print(f"Logged sklearn model to artifact path: {artifact_path}")
        print(f"  Model type: {type(model).__name__}")
        if registered_model_name:
            print(f"  Registered as: {registered_model_name}")
    
    def log_dataset(self, df: pd.DataFrame, source: str, name: str, 
                    context: str = "training", targets: Optional[str] = None):
        """
        Log a pandas DataFrame as a versioned MLflow dataset.
        
        Args:
            df: Pandas DataFrame to log
            source: Source location of the data (file path, URL, etc.)
            name: Name for the dataset
            context: Context in which dataset is used (e.g., 'training', 'validation', 'test')
            targets: Optional column name(s) for target variable(s)
        
        Returns:
            Dataset object with version information
        """
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Create MLflow dataset from pandas DataFrame
        dataset = mlflow.data.from_pandas(
            df,
            source=source,
            name=name,
            targets=targets
        )
        
        # Log the dataset with context
        mlflow.log_input(dataset, context=context)
        
        print(f"Logged dataset '{name}' (context: {context}, rows: {len(df)}, cols: {len(df.columns)})")
        print(f"  Dataset digest: {dataset.digest if hasattr(dataset, 'digest') else 'N/A'}")
        
        return dataset
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if mlflow.active_run() is None:
            print("No active MLflow run to end")
            return
        
        mlflow.end_run(status=status)
        print(f"Ended MLflow run: {self.run_id} (Status: {status})")
        self.run_id = None
        self.run_name = None
    
    def get_run_id(self) -> Optional[str]:
        """
        Get the current run ID.
        
        Returns:
            Current run ID or None if no active run
        """
        return self.run_id
    
    def get_experiment_id(self) -> str:
        """
        Get the experiment ID.
        
        Returns:
            Experiment ID
        """
        return self.experiment_id
    
    @staticmethod
    def load_sklearn_pipeline(model_uri: str) -> Pipeline:
        """
        Load a scikit-learn Pipeline from MLflow with type safety.
        
        Args:
            model_uri: MLflow model URI (e.g., 'runs:/<run_id>/pipeline' or 'models:/name/version')
        
        Returns:
            Loaded sklearn Pipeline object
        
        Raises:
            TypeError: If loaded object is not a Pipeline
        
        Examples:
            >>> pipeline = MLFlowLogger.load_sklearn_pipeline("runs:/abc123/preprocessing_pipeline")
            >>> pipeline = MLFlowLogger.load_sklearn_pipeline("models:/my_pipeline/Production")
        """
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        if not isinstance(loaded_model, Pipeline):
            raise TypeError(
                f"Expected Pipeline from '{model_uri}', got {type(loaded_model).__name__}. "
                f"Use load_sklearn_model() for non-pipeline models."
            )
        
        print(f"Loaded sklearn Pipeline from: {model_uri}")
        print(f"  Pipeline steps: {[step[0] for step in loaded_model.steps]}")
        
        return loaded_model
    
    @staticmethod
    def load_sklearn_model(model_uri: str) -> BaseEstimator:
        """
        Load a scikit-learn model (classifier/regressor) from MLflow with type safety.
        
        Args:
            model_uri: MLflow model URI (e.g., 'runs:/<run_id>/model' or 'models:/name/version')
        
        Returns:
            Loaded sklearn estimator/model object
        
        Raises:
            TypeError: If loaded object is a Pipeline (use load_sklearn_pipeline instead)
        
        Examples:
            >>> model = MLFlowLogger.load_sklearn_model("runs:/abc123/trained_model")
            >>> model = MLFlowLogger.load_sklearn_model("models:/my_classifier/Production")
        """
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        if isinstance(loaded_model, Pipeline):
            raise TypeError(
                f"Loaded object from '{model_uri}' is a Pipeline. "
                f"Use load_sklearn_pipeline() for Pipeline objects."
            )
        
        if not isinstance(loaded_model, BaseEstimator):
            raise TypeError(
                f"Expected sklearn BaseEstimator from '{model_uri}', got {type(loaded_model).__name__}"
            )
        
        print(f"Loaded sklearn model from: {model_uri}")
        print(f"  Model type: {type(loaded_model).__name__}")
        
        return loaded_model
    
    def log_keras_model(
        self, 
        model, 
        artifact_path: str = "model",
        signature=None,
        input_example=None,
        **kwargs
    ):
        """
        Log a Keras/TensorFlow model to the current MLflow run.
        
        Args:
            model: Trained Keras model to log
            artifact_path: Path within the run to save the model (default: "model")
            signature: MLflow ModelSignature (optional, recommended for inference)
            input_example: Example input for model inference documentation
            **kwargs: Additional arguments passed to mlflow.keras.log_model
        
        Returns:
            None
            
        Raises:
            RuntimeError: If no active MLflow run exists
            
        Examples:
            >>> logger.start_run(run_name="train_lstm")
            >>> logger.log_keras_model(model, "trained_model", signature=signature)
            >>> logger.end_run()
        """
        import mlflow.keras
        
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.keras.log_model(
            model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            **kwargs
        )
        
        print(f"Logged Keras model to MLflow: {artifact_path}")
        print(f"  Run ID: {self.run_id}")
        print(f"  Model type: {type(model).__name__}")
    
    def register_model(
        self, 
        model_uri: str, 
        model_name: str,
        tags: Optional[Dict[str, Any]] = None,
        await_creation: bool = True
    ):
        """
        Register a model to the MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register (e.g., 'runs:/<run_id>/model')
            model_name: Name to register the model under in the registry
            tags: Optional tags to apply to the registered model version
            await_creation: Whether to wait for model version creation (default: True)
        
        Returns:
            ModelVersion object from MLflow
            
        Examples:
            >>> run_id = logger.get_run_id()
            >>> model_uri = f"runs:/{run_id}/model"
            >>> logger.register_model(model_uri, "ml_pipeline_model", tags={"task": "regression"})
        """
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags,
            await_registration_for=30 if await_creation else 0
        )
        
        print(f"Registered model to MLflow Registry:")
        print(f"  Model Name: {model_name}")
        print(f"  Version: {result.version}")
        print(f"  Model URI: {model_uri}")
        
        return result
    
    @staticmethod
    def load_keras_model(model_uri: str):
        """
        Load a Keras/TensorFlow model from MLflow.
        
        Args:
            model_uri: MLflow model URI (e.g., 'runs:/<run_id>/model' or 'models:/name/version')
        
        Returns:
            Loaded Keras model object
        
        Examples:
            >>> model = MLFlowLogger.load_keras_model("runs:/abc123/trained_model")
            >>> model = MLFlowLogger.load_keras_model("models:/ml_pipeline_model/Production")
        """
        import mlflow.keras
        
        loaded_model = mlflow.keras.load_model(model_uri)
        
        print(f"Loaded Keras model from: {model_uri}")
        print(f"  Model type: {type(loaded_model).__name__}")
        
        return loaded_model
