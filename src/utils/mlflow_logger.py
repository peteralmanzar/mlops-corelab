"""
MLflow Logger Utility

Provides a wrapper class for MLflow tracking operations including
experiment setup, run management, and logging of metrics, parameters,
and artifacts for data exploration and model training tasks.
"""

import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path


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
