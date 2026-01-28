"""
03_dag_model - Model Training DAG

This DAG handles model training operations for the ML pipeline:
- Loads preprocessed data from 02_dag_preprocess
- Auto-detects task type (regression vs binary/multi-class classification)
- Instantiates appropriate model template from config
- Trains models on all splits/folds in parallel
- Logs comprehensive metrics, parameters, and models to MLflow
- Registers the best model to MLflow Model Registry
"""
from datetime import datetime
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from airflow.decorators import task
from airflow.models import XCom
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger
from model_template import (
    GetModelTemplateMLPRegression,
    GetModelTemplateMLPBinaryClassification,
    GetModelTemplateMLPMultiClassification,
    Optimizer
)

# Import assets
from assets import TRANSFORMED_DATA_ASSET, TRAINED_MODEL_ASSET

# Load configuration
config = Config.load()


def detect_task_type(y: pd.DataFrame, label_cols: List[str]) -> Tuple[str, int]:
    """
    Auto-detect task type from label data.
    
    Returns:
        tuple: (task_type, num_classes)
            task_type: 'regression', 'binary_classification', or 'multi_classification'
            num_classes: Number of classes (1 for regression, 2 for binary, >2 for multi)
    """
    # Get the first label column for analysis
    label_col = label_cols[0]
    
    if label_col not in y.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available: {y.columns.tolist()}")
    
    y_values = y[label_col].dropna()
    
    # Check if continuous (regression) or discrete (classification)
    unique_values = y_values.nunique()
    
    # Heuristic: if fewer than 20 unique values and all are integers, assume classification
    if unique_values < 20 and all(y_values == y_values.astype(int)):
        if unique_values == 2:
            task_type = "binary_classification"
            num_classes = 2
        else:
            task_type = "multi_classification"
            num_classes = unique_values
    else:
        task_type = "regression"
        num_classes = 1
    
    print(f"Auto-detected task type: {task_type}")
    print(f"  Unique values: {unique_values}")
    print(f"  Value range: [{y_values.min()}, {y_values.max()}]")
    print(f"  Number of classes: {num_classes}")
    
    return task_type, num_classes


def metadata_load(**context):
    """
    Load preprocessed data metadata from 02_dag_preprocess XCom.
    Returns information about split type and file paths.
    """
    ti = context['ti']
    
    # Pull metadata from previous DAG run (02_dag_preprocess)
    # include_prior_dates=True returns a list, so we take the first (most recent) value
    split_metadata_list = ti.xcom_pull(dag_id='02_dag_preprocess', task_ids='metadata_load', key='split_metadata', include_prior_dates=True)
    
    if split_metadata_list is None or len(split_metadata_list) == 0:
        raise ValueError("No split metadata found from 02_dag_preprocess. Ensure 02_dag_preprocess has run successfully.")
    
    # Extract the most recent metadata (first item in the list)
    split_metadata = split_metadata_list[0] if isinstance(split_metadata_list, list) else split_metadata_list
    
    print(f"DEBUG: Full split_metadata = {split_metadata}")
    print(f"DEBUG: Type of split_metadata = {type(split_metadata)}")
    
    # Extract split_type - handle case where it might be a list itself
    split_type_value = split_metadata.get('split_type')
    if isinstance(split_type_value, list):
        # If it's a list, take the first element
        split_type = split_type_value[0]
    else:
        split_type = split_type_value
    
    print(f"Split type detected: {split_type}")
    
    # Get features path
    features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")
    
    # Prepare fold information
    folds_info = []
    
    if split_type == 'simple':
        # Single train/test split
        folds_info.append({
            'fold_id': 0,
            'train_path': os.path.join(features_path, "train_transformed.csv"),
            'test_path': os.path.join(features_path, "test_transformed.csv")
        })
    else:
        # K-Fold/Stratified/TimeSeries splits
        num_folds_value = split_metadata.get('num_folds', 5)
        # Handle case where num_folds might be a list
        num_folds = num_folds_value[0] if isinstance(num_folds_value, list) else num_folds_value
        
        for fold_idx in range(num_folds):
            folds_info.append({
                'fold_id': fold_idx,
                'train_path': os.path.join(features_path, f"train_fold_{fold_idx}_transformed.csv"),
                'test_path': os.path.join(features_path, f"test_fold_{fold_idx}_transformed.csv")
            })
    
    # Pull pipeline_run_id from first DAG for traceability
    pipeline_run_id_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='raw_data_load', key='pipeline_run_id', include_prior_dates=True)
    pipeline_run_id = pipeline_run_id_list[0] if isinstance(pipeline_run_id_list, list) and pipeline_run_id_list else None
    
    if pipeline_run_id:
        print(f"Pipeline Run ID: {pipeline_run_id}")
    else:
        print("Warning: No pipeline_run_id found from 01_dag_data")
    
    # Push metadata for downstream tasks
    context['ti'].xcom_push(key='pipeline_run_id', value=pipeline_run_id)
    context['ti'].xcom_push(key='split_type', value=split_type)
    context['ti'].xcom_push(key='folds_info', value=folds_info)
    context['ti'].xcom_push(key='num_folds', value=len(folds_info))
    
    # Validate that files exist
    for fold_info in folds_info:
        train_path = fold_info['train_path']
        test_path = fold_info['test_path']
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Train file not found: {train_path}\n"
                f"Split type is '{split_type}' but expected files are missing.\n"
                f"This might indicate a mismatch between the data split configuration and preprocessing output.\n"
                f"Please re-run 01_dag_data and 02_dag_preprocess with consistent configuration."
            )
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test file not found: {test_path}\n"
                f"Split type is '{split_type}' but expected files are missing.\n"
                f"This might indicate a mismatch between the data split configuration and preprocessing output.\n"
                f"Please re-run 01_dag_data and 02_dag_preprocess with consistent configuration."
            )
    
    print(f"Loaded metadata for {len(folds_info)} fold(s)")
    print(f"All expected files validated successfully")
    
    # Return folds_info for dynamic task mapping
    return folds_info


def model_build(**context):
    """
    Build model configuration by auto-detecting task type and preparing model template parameters.
    Reads first fold to infer input shape and task type.
    """
    ti = context['ti']
    
    # Get fold information
    folds_info = ti.xcom_pull(task_ids='metadata_load', key='folds_info')
    
    if not folds_info:
        raise ValueError("No folds information found from metadata_load task")
    
    # Load first training fold to infer shape and task type
    first_fold = folds_info[0]
    train_path = first_fold['train_path']
    
    print(f"Loading data from: {train_path}")
    df = pd.read_csv(train_path)
    
    # Separate features and labels
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    X = df[feature_cols]
    y = df[label_cols]
    
    # Infer shape
    num_features = X.shape[1]
    num_samples = X.shape[0]
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {num_features} columns")
    print(f"Samples: {num_samples} rows")
    print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
    print(f"Label columns: {label_cols}")
    
    # Auto-detect task type
    task_type, num_classes = detect_task_type(y, label_cols)
    
    # Get optimizer from config
    optimizer_str = config.MODEL.get("OPTIMIZER", "adam").lower()
    optimizer_map = {
        "adam": Optimizer.ADAM,
        "sgd": Optimizer.SGD,
        "rmsprop": Optimizer.RMS_PROP,
        "adadelta": Optimizer.ADA_DELTA,
        "adagrad": Optimizer.ADA_GRAD,
        "adamax": Optimizer.ADA_MAX,
        "nadam": Optimizer.NADAM,
        "ftrl": Optimizer.FTRL
    }
    optimizer = optimizer_map.get(optimizer_str, Optimizer.ADAM)
    
    # Prepare model configuration
    model_config = {
        'task_type': task_type,
        'num_classes': num_classes,
        'num_features': num_features,
        'optimizer': optimizer,
        'optimizer_name': optimizer_str,
        'epochs': config.MODEL.get("EPOCHS", 50),
        'batch_size': config.MODEL.get("BATCH_SIZE", 32),
        'validation_split': config.MODEL.get("VALIDATION_SPLIT", 0.2),
        'early_stopping_patience': config.MODEL.get("EARLY_STOPPING_PATIENCE", 10)
    }
    
    # Push model config to XCom (serialize enum to string)
    model_config_serializable = {k: (v.value if hasattr(v, 'value') else v) for k, v in model_config.items()}
    context['ti'].xcom_push(key='model_config', value=model_config_serializable)
    
    print(f"\nModel Configuration:")
    print(json.dumps(model_config_serializable, indent=2))
    
    # Return serializable model_config for dynamic task mapping
    return model_config_serializable


@task
def train_fold_model(fold_info: dict, model_config: dict) -> dict:
    """
    Train model for a single fold and log to MLflow.
    
    Args:
        fold_info: Dictionary with fold_id, train_path, test_path
        model_config: Model configuration dictionary
    
    Returns:
        Dictionary with fold results including metrics and run_id
    """
    fold_id = fold_info['fold_id']
    train_path = fold_info['train_path']
    test_path = fold_info['test_path']
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_id}")
    print(f"{'='*60}")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and labels
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    feature_cols = [col for col in train_df.columns if col not in label_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[label_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_cols].values
    
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    
    # Instantiate model based on task type
    task_type = model_config['task_type']
    num_features = model_config['num_features']
    num_classes = model_config['num_classes']
    
    # Reconstruct optimizer enum from serialized value
    optimizer_value = model_config['optimizer']
    if isinstance(optimizer_value, str):
        # If it's a string, convert back to enum
        optimizer = Optimizer(optimizer_value)
    else:
        # If it's already an enum (shouldn't happen but safe)
        optimizer = optimizer_value
    
    if task_type == 'regression':
        model = GetModelTemplateMLPRegression(numberOfFeatures=num_features, optimizer=optimizer)
        metrics_to_track = ['loss', 'mean_squared_error']
    elif task_type == 'binary_classification':
        model = GetModelTemplateMLPBinaryClassification(numberOfFeatures=num_features, optimizer=optimizer)
        metrics_to_track = ['loss', 'accuracy']
    elif task_type == 'multi_classification':
        model = GetModelTemplateMLPMultiClassification(
            numberOfFeatures=num_features,
            num_classes=num_classes,
            optimizer=optimizer
        )
        metrics_to_track = ['loss', 'accuracy']
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print(f"\nModel instantiated: {task_type}")
    print(f"Model summary:")
    model.summary()
    
    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Get Airflow context for TaskFlow function
    context = get_current_context()
    ti = context['ti']
    
    # Get pipeline run ID for traceability
    pipeline_run_id = ti.xcom_pull(task_ids='metadata_load', key='pipeline_run_id')
    dag_run_id = context.get('dag_run').run_id
    
    # Start MLflow run
    run_name = f"{pipeline_run_id}_Model_Train_Fold_{fold_id}"
    tags = {
        "task_type": "model_training",
        "model_type": task_type,
        "fold_id": str(fold_id),
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "airflow_dag_run_id": dag_run_id,
        "pipeline_run_id": pipeline_run_id if pipeline_run_id else 'unknown'
    }
    
    try:
        logger.start_run(run_name=run_name, tags=tags)
        
        # Log parameters
        params = {
            "fold_id": fold_id,
            "task_type": task_type,
            "num_features": num_features,
            "num_classes": num_classes,
            "optimizer": model_config['optimizer_name'],
            "epochs": model_config['epochs'],
            "batch_size": model_config['batch_size'],
            "validation_split": model_config['validation_split'],
            "early_stopping_patience": model_config['early_stopping_patience'],
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        logger.log_params(params)
        
        # Setup early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=model_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        print(f"\nTraining model for {model_config['epochs']} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_split=model_config['validation_split'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        
        # Log metrics
        final_metrics = {}
        
        # Training history metrics (final epoch)
        for metric_name in history.history.keys():
            final_value = history.history[metric_name][-1]
            final_metrics[f"final_{metric_name}"] = float(final_value)
        
        # Test metrics - handle both old and new Keras metric naming
        test_metric_names = model.metrics_names
        for i, metric_name in enumerate(test_metric_names):
            final_metrics[f"test_{metric_name}"] = float(test_results[i])
        
        # Ensure test_accuracy is explicitly set for classification
        if task_type in ['binary_classification', 'multi_classification']:
            # If we have multiple test results and the second one looks like accuracy
            if len(test_results) > 1:
                # The second metric is typically accuracy in Keras
                if 'test_accuracy' not in final_metrics:
                    final_metrics['test_accuracy'] = float(test_results[1])
                    print(f"  Explicitly set test_accuracy: {test_results[1]:.4f}")
        
        # Additional metrics
        final_metrics['total_epochs_trained'] = len(history.history['loss'])
        final_metrics['stopped_early'] = int(len(history.history['loss']) < model_config['epochs'])
        
        logger.log_metrics(final_metrics)
        
        # Log model
        print(f"\nLogging model to MLflow...")
        logger.log_keras_model(
            model=model,
            artifact_path="model"
        )
        
        # Log training history as artifact
        history_df = pd.DataFrame(history.history)
        logger.log_dataframe(history_df, "training_history.csv")
        
        # End MLflow run
        run_id = logger.get_run_id()
        logger.end_run(status="FINISHED")
        
        # Prepare results
        results = {
            'fold_id': fold_id,
            'run_id': run_id,
            'task_type': task_type,
            'metrics': final_metrics,
            'model_uri': f"runs:/{run_id}/model"
        }
        
        print(f"\nFold {fold_id} training completed successfully!")
        print(f"Run ID: {run_id}")
        print(f"Test {test_metric_names[0]}: {test_results[0]:.4f}")
        if len(test_results) > 1:
            print(f"Test {test_metric_names[1]}: {test_results[1]:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error during training fold {fold_id}: {e}")
        logger.end_run(status="FAILED")
        raise


def model_register(**context):
    """
    Aggregate fold results, identify best model, and register to MLflow Model Registry.
    """
    ti = context['ti']
    
    # Get fold results from train_fold_model task
    fold_results = ti.xcom_pull(task_ids='train_fold_model')
    
    if not fold_results:
        raise ValueError("No fold results found from train_fold_model task")
    
    # Handle both single result (dict) and multiple results (list of dicts)
    if isinstance(fold_results, dict):
        fold_results = [fold_results]
    
    print(f"\n{'='*60}")
    print(f"Model Registration - Analyzing {len(fold_results)} fold(s)")
    print(f"{'='*60}")
    
    # Debug: print available metrics
    if fold_results:
        print(f"\nAvailable metrics in fold results:")
        print(f"  Metrics keys: {list(fold_results[0]['metrics'].keys())}")
    
    # Determine best metric based on task type
    task_type = fold_results[0]['task_type']
    
    if task_type == 'regression':
        # For regression, lower loss is better
        best_metric_key = 'test_loss'
        best_fold = min(fold_results, key=lambda x: x['metrics'].get(best_metric_key, float('inf')))
    else:
        # For classification, higher accuracy is better
        # Try test_accuracy first, fall back to final_accuracy if not available
        test_acc_key = 'test_accuracy'
        final_acc_key = 'final_accuracy'
        
        # Check which metric is available
        sample_metrics = fold_results[0]['metrics']
        if test_acc_key in sample_metrics:
            best_metric_key = test_acc_key
        elif final_acc_key in sample_metrics:
            best_metric_key = final_acc_key
            print(f"Note: Using {final_acc_key} instead of {test_acc_key}")
        else:
            # Last resort: use test_loss (lower is better)
            best_metric_key = 'test_loss'
            print(f"Warning: No accuracy metric found, using test_loss instead")
        
        # Get best fold based on metric type
        if 'accuracy' in best_metric_key:
            best_fold = max(fold_results, key=lambda x: x['metrics'].get(best_metric_key, 0.0))
        else:
            # Using loss, lower is better
            best_fold = min(fold_results, key=lambda x: x['metrics'].get(best_metric_key, float('inf')))
    
    print(f"\nBest model selection metric: {best_metric_key}")
    print(f"Best fold: {best_fold['fold_id']}")
    
    # Get metric value safely
    best_metric_value = best_fold['metrics'].get(best_metric_key)
    if best_metric_value is not None:
        print(f"Best {best_metric_key}: {best_metric_value:.4f}")
    else:
        print(f"Warning: {best_metric_key} not found in metrics. Available: {list(best_fold['metrics'].keys())}")
        # Raise error if metric not found
        raise KeyError(f"Required metric '{best_metric_key}' not found in fold results. Available metrics: {list(best_fold['metrics'].keys())}")
    
    # Print all fold results
    print(f"\nAll fold results:")
    for result in fold_results:
        fold_id = result['fold_id']
        metric_val = result['metrics'].get(best_metric_key, 'N/A')
        if metric_val != 'N/A' and isinstance(metric_val, (int, float)):
            print(f"  Fold {fold_id}: {best_metric_key}={metric_val:.4f}")
        else:
            print(f"  Fold {fold_id}: {best_metric_key}={metric_val}")
    
    # Initialize MLflow logger for registration
    mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Register best model
    model_uri = best_fold['model_uri']
    model_name = "ml_pipeline_model"
    
    registration_tags = {
        "task_type": task_type,
        "best_fold_id": str(best_fold['fold_id']),
        "best_metric": best_metric_key,
        "best_metric_value": str(best_metric_value),
        "registered_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\nRegistering model to MLflow Model Registry...")
    print(f"  Model URI: {model_uri}")
    print(f"  Model Name: {model_name}")
    
    registered_model = logger.register_model(
        model_uri=model_uri,
        model_name=model_name,
        tags=registration_tags
    )
    
    # Push registration info to XCom
    registration_info = {
        'model_name': model_name,
        'model_version': registered_model.version,
        'model_uri': model_uri,
        'best_fold_id': best_fold['fold_id'],
        'best_run_id': best_fold['run_id'],
        'best_metric': best_metric_key,
        'best_metric_value': best_metric_value,
        'task_type': task_type,
        'all_fold_results': fold_results
    }
    
    context['ti'].xcom_push(key='registration_info', value=registration_info)
    
    print(f"\n{'='*60}")
    print(f"Model registered successfully!")
    print(f"  Version: {registered_model.version}")
    print(f"  Best fold: {best_fold['fold_id']}")
    print(f"  {best_metric_key}: {best_metric_value:.4f}")
    print(f"{'='*60}")
    
    return f"Registered {model_name} v{registered_model.version} from fold {best_fold['fold_id']}"


# DAG definition
with DAG(
    dag_id="03_dag_model",
    default_args=config.DEFAULT_DAG_ARGS,
    description="Model training DAG - auto-detects task type, trains models on all folds in parallel, logs to MLflow, and registers best model to Model Registry",
    schedule=[TRANSFORMED_DATA_ASSET],  # Trigger when 02_dag_preprocess completes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["model-training", "ml-pipeline"],
) as dag:
    
    # Task 1: Load preprocessed data metadata from DAG 2
    metadata_load_task = PythonOperator(
        task_id="metadata_load",
        python_callable=metadata_load,
    )
    
    # Task 2: Build model configuration (auto-detect task type, infer shapes)
    model_build_task = PythonOperator(
        task_id="model_build",
        python_callable=model_build,
    )
    
    # Task 3: Train models for all folds in parallel using dynamic task mapping
    train_fold_model_tasks = train_fold_model.partial(
        model_config=model_build_task.output
    ).expand(
        fold_info=metadata_load_task.output
    )
    train_fold_model_tasks.operator.outlets = [TRAINED_MODEL_ASSET]  # This task produces the trained model
    
    # Task 4: Register best model to MLflow Model Registry
    model_register_task = PythonOperator(
        task_id="model_register",
        python_callable=model_register,
    )
    
    # Task dependencies
    metadata_load_task >> model_build_task >> train_fold_model_tasks >> model_register_task
