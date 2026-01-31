"""
02_dag_preprocess - Data Preprocessing DAG

This DAG handles preprocessing operations for the ML pipeline:
- Loads split/fold data from 01_dag_data
- Builds and saves a global preprocessing pipeline
- Transforms data through the pipeline (parallel for folds)
- Performs EDA on preprocessed features with correlation analysis
- Logs all operations to MLflow
"""
from datetime import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.decorators import task

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger
from data_pipeline import build_pipeline, save_pipeline, load_pipeline

# Import assets
from assets import TRAIN_TEST_SPLIT_ASSET, PREPROCESSING_PIPELINE_ASSET, TRANSFORMED_DATA_ASSET

# Load configuration
config = Config.load()


def metadata_load(**context):
    """
    Load split metadata by reading config directly and detecting available files.
    This makes the DAG robust to config changes between runs.
    Falls back to XCom if needed for backward compatibility.
    """
    ti = context['ti']

    # Read split type directly from config (source of truth)
    fold_type = config.DATA.get("FOLD_TYPE")
    split_type = fold_type if fold_type else "simple"

    print(f"Split type from config: {split_type}")

    # Verify files exist for this split type
    processed_path = config.DATA.get("PROCESSED_PATH")

    if split_type == "simple":
        # Check for simple split files
        train_file = os.path.join(processed_path, "train.csv")
        test_file = os.path.join(processed_path, "test.csv")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(
                f"Simple split files not found in {processed_path}. "
                f"Expected: train.csv, test.csv. "
                f"Please run 01_dag_data with FOLD_TYPE=null/simple in config."
            )
    else:
        # Check for fold files
        import glob
        fold_train_files = glob.glob(os.path.join(processed_path, "train_fold_*.csv"))
        if not fold_train_files:
            raise FileNotFoundError(
                f"No fold files found in {processed_path}. "
                f"Expected: train_fold_*.csv files. "
                f"Config shows FOLD_TYPE={fold_type}. "
                f"Please run 01_dag_data to generate {fold_type} splits."
            )
        print(f"Found {len(fold_train_files)} fold files")

    print(f"Split type verified: {split_type}")
    
    # Build metadata from actual files on disk
    metadata = {
        'split_type': split_type
    }

    if split_type == 'simple':
        # Simple split - read file info directly
        train_path = os.path.join(processed_path, "train.csv")
        test_path = os.path.join(processed_path, "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        metadata['train_path'] = train_path
        metadata['test_path'] = test_path
        metadata['train_shape'] = train_df.shape
        metadata['test_shape'] = test_df.shape
        print(f"Simple split: train={metadata['train_shape']}, test={metadata['test_shape']}")

    elif split_type in ['kfold', 'stratified', 'timeseries']:
        # Fold-based split - auto-detect fold files
        import glob

        # Use val_fold files for validation (test equivalent in cross-validation)
        train_files = sorted(glob.glob(os.path.join(processed_path, "train_fold_*.csv")))
        val_files = sorted(glob.glob(os.path.join(processed_path, "val_fold_*.csv")))

        if len(train_files) == 0:
            raise FileNotFoundError(f"No train_fold_*.csv files found in {processed_path}")
        if len(val_files) == 0:
            raise FileNotFoundError(f"No val_fold_*.csv files found in {processed_path}")

        num_folds = len(train_files)
        fold_paths = []

        for i in range(num_folds):
            # Extract fold index from filename
            train_fold_path = train_files[i]
            val_fold_path = val_files[i]

            fold_paths.append({
                'fold': i,
                'train': train_fold_path,
                'test': val_fold_path  # Use 'test' key for consistency with model training
            })

        metadata['num_folds'] = num_folds
        metadata['fold_paths'] = fold_paths
        print(f"{split_type} split: {num_folds} folds detected")

        if split_type == 'timeseries':
            metadata['ts_gap'] = config.DATA.get("TIME_SERIES_GAP", 0)
            metadata['ts_expanding'] = config.DATA.get("TIME_SERIES_EXPANDING", True)
    
    # Get the triggering asset events to find the correct upstream DAG run
    triggering_events = context.get('triggering_asset_events')
    upstream_run_id = None

    if triggering_events:
        print(f"DEBUG: triggering_asset_events = {triggering_events}")
        for asset_uri, events in triggering_events.items():
            print(f"DEBUG: Asset {asset_uri} has {len(events)} events")
            for event in events:
                print(f"DEBUG: Event: {event}")
                if hasattr(event, 'source_run_id'):
                    upstream_run_id = event.source_run_id
                    print(f"DEBUG: Found upstream_run_id from event.source_run_id: {upstream_run_id}")
                    break
                elif hasattr(event, 'extra') and event.extra:
                    upstream_run_id = event.extra.get('run_id')
                    print(f"DEBUG: Found upstream_run_id from event.extra: {upstream_run_id}")
                    break
            if upstream_run_id:
                break

    # Pull invocation_id using specific run_id if available
    invocation_id = None
    if upstream_run_id:
        print(f"DEBUG: Pulling XCom with run_id={upstream_run_id}")
        invocation_id = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='invocation_id', run_id=upstream_run_id)
        print(f"DEBUG: invocation_id from specific run = {invocation_id}")

    # Fallback: use include_prior_dates if we couldn't get the specific run
    if not invocation_id:
        invocation_id_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='invocation_id', include_prior_dates=True)
        print(f"DEBUG: invocation_id_list (fallback) = {invocation_id_list}, type = {type(invocation_id_list)}")
        if invocation_id_list is not None:
            if isinstance(invocation_id_list, list) and len(invocation_id_list) > 0:
                invocation_id = invocation_id_list[0]  # Take first (most recent)
            elif isinstance(invocation_id_list, str):
                invocation_id = invocation_id_list

    if invocation_id:
        print(f"Pipeline Tag: {invocation_id}")
        metadata['invocation_id'] = invocation_id
    else:
        print("ERROR: No invocation_id found from 01_dag_data!")
        print("This usually means DAG 01 did not run successfully or XCom data is missing.")
        raise ValueError("invocation_id not found from 01_dag_data. Ensure 01_dag_data completed successfully.")

    # Push metadata for downstream tasks
    context['ti'].xcom_push(key='split_metadata', value=metadata)
    context['ti'].xcom_push(key='invocation_id', value=invocation_id)

    return f"Loaded {split_type} split metadata (Pipeline Tag: {invocation_id})"  


def pipeline_build(**context):
    """
    Build preprocessing pipeline from config and save it.
    Uses a global pipeline that will be applied to all splits/folds.
    Logs the pipeline to MLflow as an artifact.
    """
    print("Building preprocessing pipeline from config...")
    
    # Build pipeline using config
    pipeline = build_pipeline(config)
    
    # Get pipeline save path
    artifacts_path = config.MODEL.get("ARTIFACTS_PATH", "/mlflow/artifacts")
    os.makedirs(artifacts_path, exist_ok=True)
    pipeline_path = os.path.join(artifacts_path, "preprocessing_pipeline.joblib")
    
    # Load training data to fit the pipeline
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
    
    print(f"Split metadata: {metadata}")
    
    if metadata['split_type'] == 'simple':
        train_path = metadata['train_path']
    else:
        # For folds, use the first fold's training data to fit
        train_path = metadata['fold_paths'][0]['train']
    
    print(f"Fitting pipeline on: {train_path}")
    train_df = pd.read_csv(train_path)
    
    # Separate features and labels
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    feature_cols = [col for col in train_df.columns if col not in label_cols]
    X_train = train_df[feature_cols]
    
    # Fit pipeline
    print(f"Original features shape: {X_train.shape}")
    print(f"Original features: {X_train.columns.tolist()}")
    
    pipeline.fit(X_train)
    
    # Test transform to get output shape
    X_transformed = pipeline.transform(X_train)
    print(f"Transformed features shape: {X_transformed.shape}")
    if hasattr(X_transformed, 'columns'):
        print(f"Transformed features: {X_transformed.columns.tolist()}")
    
    # Save pipeline to file
    save_pipeline(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")
    
    # Initialize MLflow logger and log pipeline
    mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Get pipeline tag for traceability
    invocation_id = metadata.get('invocation_id', 'unknown')
    dag_run_id = context.get('dag_run').run_id

    # Start MLflow run
    run_name = "D2S1_Preprocess_Pipeline_Build"
    tags = {
        "task_type": "preprocessing_pipeline",
        "split_type": metadata['split_type'],
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date')),
        "airflow_dag_run_id": dag_run_id,
        "invocation_id": invocation_id,
        "pipeline_step": "D2S1"
    }
    
    try:
        logger.start_run(run_name=run_name, tags=tags)
        
        # Log training dataset version
        logger.log_dataset(
            df=train_df,
            source=train_path,
            name="pipeline_training_data",
            context="pipeline_fit",
            targets=",".join(label_cols)
        )
        
        # Log pipeline parameters
        pipeline_params = {
            "n_steps": len(pipeline.steps),
            "step_names": ",".join([step[0] for step in pipeline.steps]),
            "input_features": len(feature_cols),
            "original_shape": str(X_train.shape),
            "transformed_shape": str(X_transformed.shape),
            "training_data_path": train_path
        }
        logger.log_params(pipeline_params)
        
        # Log pipeline metrics
        metrics = {
            "n_pipeline_steps": len(pipeline.steps),
            "n_input_features": X_train.shape[1],
            "n_output_features": X_transformed.shape[1] if hasattr(X_transformed, 'shape') else len(X_transformed.columns),
            "dimensionality_reduction_ratio": round(X_train.shape[1] / (X_transformed.shape[1] if hasattr(X_transformed, 'shape') else len(X_transformed.columns)), 4)
        }
        logger.log_metrics(metrics)
        
        # Log the sklearn pipeline as an MLflow model
        logger.log_sklearn_pipeline(
            pipeline=pipeline,
            artifact_path="preprocessing_pipeline",
            input_example=X_train.head(5) if len(X_train) >= 5 else X_train
        )
        
        # Also log the pipeline file directly as an artifact
        logger.log_artifact(pipeline_path, "pipeline_artifacts")
        
        # Log pipeline step details as text
        pipeline_details = []
        pipeline_details.append("Preprocessing Pipeline Steps:")
        pipeline_details.append("=" * 60)
        for i, (step_name, step_obj) in enumerate(pipeline.steps):
            pipeline_details.append(f"\nStep {i+1}: {step_name}")
            pipeline_details.append(f"  Type: {type(step_obj).__name__}")
            if hasattr(step_obj, 'get_params'):
                params = step_obj.get_params()
                for param_name, param_value in params.items():
                    if not param_name.startswith('_'):
                        pipeline_details.append(f"  {param_name}: {param_value}")
        pipeline_details.append("=" * 60)
        
        logger.log_text("\n".join(pipeline_details), "pipeline_details.txt")
        
        # Capture MLflow run id and push to XCom for downstream DAGs
        run_id = logger.get_run_id()
        context['ti'].xcom_push(key='mlflow_run_id', value=run_id)
        context['ti'].xcom_push(key='pipeline_mlflow_run_id', value=run_id)

        # End MLflow run
        logger.end_run(status="FINISHED")

        print(f"Pipeline logged to MLflow (Run ID: {run_id})")
        
    except Exception as e:
        print(f"Error logging pipeline to MLflow: {e}")
        logger.end_run(status="FAILED")
        # Don't raise - pipeline is still saved to disk
    
    # Push pipeline path to XCom
    context['ti'].xcom_push(key='pipeline_path', value=pipeline_path)
    context['ti'].xcom_push(key='original_shape', value=X_train.shape)
    context['ti'].xcom_push(key='transformed_shape', value=X_transformed.shape)
    
    return f"Pipeline built and saved: {pipeline_path}"


@task
def split_transform(split_info: dict):
    """
    Transform a single train/test split using the pipeline.
    This task is designed for dynamic task mapping (parallel execution).
    
    Args:
        split_info: Dictionary with 'train', 'test', 'fold', 'pipeline_path', 'features_path', 'label_cols' keys
    """
    import pandas as pd
    from data_pipeline import load_pipeline
    
    fold = split_info.get('fold', 'simple')
    train_path = split_info['train']
    test_path = split_info['test']
    pipeline_path = split_info['pipeline_path']
    features_path = split_info['features_path']
    label_cols = split_info['label_cols']
    
    print(f"Processing fold: {fold}")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    # Load pipeline
    pipeline = load_pipeline(pipeline_path)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and labels
    X_train = train_df[[col for col in train_df.columns if col not in label_cols]]
    y_train = train_df[label_cols] if label_cols else None
    
    X_test = test_df[[col for col in test_df.columns if col not in label_cols]]
    y_test = test_df[label_cols] if label_cols else None
    
    # Transform features
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Convert to DataFrame if numpy array
    if isinstance(X_train_transformed, np.ndarray):
        X_train_transformed = pd.DataFrame(X_train_transformed)
        X_test_transformed = pd.DataFrame(X_test_transformed)
    
    # Recombine with labels
    if y_train is not None:
        train_transformed = pd.concat([X_train_transformed, y_train.reset_index(drop=True)], axis=1)
    else:
        train_transformed = X_train_transformed
    
    if y_test is not None:
        test_transformed = pd.concat([X_test_transformed, y_test.reset_index(drop=True)], axis=1)
    else:
        test_transformed = X_test_transformed
    
    # Save transformed data
    os.makedirs(features_path, exist_ok=True)
    
    if fold == 'simple':
        train_output = os.path.join(features_path, "train_transformed.csv")
        test_output = os.path.join(features_path, "test_transformed.csv")
    else:
        train_output = os.path.join(features_path, f"train_fold_{fold}_transformed.csv")
        test_output = os.path.join(features_path, f"test_fold_{fold}_transformed.csv")
    
    train_transformed.to_csv(train_output, index=False)
    test_transformed.to_csv(test_output, index=False)
    
    print(f"  Saved: {train_output} (shape: {train_transformed.shape})")
    print(f"  Saved: {test_output} (shape: {test_transformed.shape})")
    
    return {
        'fold': fold,
        'train_output': train_output,
        'test_output': test_output,
        'train_shape': train_transformed.shape,
        'test_shape': test_transformed.shape
    }


@task
def transform_prepare(**context):
    """
    Prepare the list of splits to transform in parallel.
    Returns list of splits for dynamic task mapping.
    """
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
    pipeline_path = ti.xcom_pull(task_ids='pipeline_build', key='pipeline_path')
    
    features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    
    splits_to_process = []
    
    if metadata['split_type'] == 'simple':
        splits_to_process.append({
            'fold': 'simple',
            'train': metadata['train_path'],
            'test': metadata['test_path'],
            'pipeline_path': pipeline_path,
            'features_path': features_path,
            'label_cols': label_cols
        })
    else:
        for fold_info in metadata['fold_paths']:
            fold_info['pipeline_path'] = pipeline_path
            fold_info['features_path'] = features_path
            fold_info['label_cols'] = label_cols
            splits_to_process.append(fold_info)
    
    return splits_to_process


def validate_transformed_data(**context):
    """
    Validate transformed data for strict quality requirements:
    - Zero null values allowed
    - All feature columns must be numeric
    
    Raises ValueError if validation fails.
    Logs validation results to MLflow.
    """
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
    features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    
    print("Starting data validation...")
    print(f"Validation Rules: Zero nulls, All numeric features")
    
    # Collect all transformed file paths
    files_to_validate = []
    
    if metadata['split_type'] == 'simple':
        files_to_validate.append({
            'name': 'train',
            'path': os.path.join(features_path, "train_transformed.csv")
        })
        files_to_validate.append({
            'name': 'test',
            'path': os.path.join(features_path, "test_transformed.csv")
        })
    else:
        # Get number of folds
        num_folds = metadata.get('num_folds', 5)
        for fold_idx in range(num_folds):
            files_to_validate.append({
                'name': f'train_fold_{fold_idx}',
                'path': os.path.join(features_path, f"train_fold_{fold_idx}_transformed.csv")
            })
            files_to_validate.append({
                'name': f'test_fold_{fold_idx}',
                'path': os.path.join(features_path, f"test_fold_{fold_idx}_transformed.csv")
            })
    
    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Get pipeline tag for traceability
    invocation_id = metadata.get('invocation_id', 'unknown')
    dag_run_id = context.get('dag_run').run_id

    # Start MLflow run
    run_name = "D2S2_Preprocess_Validation"
    tags = {
        "task_type": "validation",
        "split_type": metadata['split_type'],
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date')),
        "airflow_dag_run_id": dag_run_id,
        "invocation_id": invocation_id,
        "pipeline_step": "D2S2"
    }
    
    validation_passed = True
    validation_errors = []
    validation_warnings = []
    total_null_count = 0
    total_non_numeric_features = 0
    files_validated = 0
    
    try:
        logger.start_run(run_name=run_name, tags=tags)
        
        # Validate each file
        for file_info in files_to_validate:
            file_name = file_info['name']
            file_path = file_info['path']
            
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                validation_errors.append(error_msg)
                print(f"ERROR: {error_msg}")
                validation_passed = False
                continue
            
            print(f"\nValidating: {file_name} ({file_path})")
            df = pd.read_csv(file_path)
            
            # Separate features from labels
            feature_cols = [col for col in df.columns if col not in label_cols]
            
            # Check 1: No null values
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                null_details = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
                error_msg = f"{file_name}: Found {null_count} null values in columns: {null_details}"
                validation_errors.append(error_msg)
                print(f"  ❌ NULL CHECK FAILED: {null_count} null values found")
                print(f"     Columns with nulls: {null_details}")
                validation_passed = False
                total_null_count += null_count
            else:
                print(f"  ✓ NULL CHECK PASSED: Zero null values")
            
            # Check 2: All feature columns are numeric
            non_numeric_features = df[feature_cols].select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            if len(non_numeric_features) > 0:
                non_numeric_dtypes = {col: str(df[col].dtype) for col in non_numeric_features}
                error_msg = f"{file_name}: Found {len(non_numeric_features)} non-numeric feature columns: {non_numeric_dtypes}"
                validation_errors.append(error_msg)
                print(f"  ❌ NUMERIC CHECK FAILED: {len(non_numeric_features)} non-numeric features")
                print(f"     Non-numeric features: {non_numeric_dtypes}")
                validation_passed = False
                total_non_numeric_features += len(non_numeric_features)
            else:
                print(f"  ✓ NUMERIC CHECK PASSED: All {len(feature_cols)} features are numeric")
            
            # Log per-file metrics
            logger.log_metrics({
                f"{file_name}_null_count": int(null_count),
                f"{file_name}_non_numeric_count": len(non_numeric_features),
                f"{file_name}_total_rows": len(df),
                f"{file_name}_total_features": len(feature_cols)
            })
            
            files_validated += 1
        
        # Log aggregate metrics
        logger.log_metrics({
            "validation_passed": 1 if validation_passed else 0,
            "total_null_count": total_null_count,
            "total_non_numeric_features": total_non_numeric_features,
            "files_validated": files_validated,
            "validation_errors": len(validation_errors),
            "validation_warnings": len(validation_warnings)
        })
        
        # Log validation parameters
        logger.log_params({
            "validation_rule_nulls": "zero_allowed",
            "validation_rule_types": "all_numeric_features",
            "split_type": metadata['split_type'],
            "files_checked": files_validated
        })
        
        # Create validation report
        report_lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Split Type: {metadata['split_type']}",
            f"Files Validated: {files_validated}",
            "",
            "VALIDATION RULES:",
            "  1. Zero null values allowed",
            "  2. All feature columns must be numeric (int64, float64, int32, float32)",
            "",
            f"VALIDATION STATUS: {'✓ PASSED' if validation_passed else '✗ FAILED'}",
            "",
            "SUMMARY:",
            f"  - Total Null Values: {total_null_count}",
            f"  - Non-Numeric Features: {total_non_numeric_features}",
            f"  - Errors: {len(validation_errors)}",
            f"  - Warnings: {len(validation_warnings)}",
            ""
        ]
        
        if validation_errors:
            report_lines.append("ERRORS:")
            for i, error in enumerate(validation_errors, 1):
                report_lines.append(f"  {i}. {error}")
            report_lines.append("")
        
        if validation_warnings:
            report_lines.append("WARNINGS:")
            for i, warning in enumerate(validation_warnings, 1):
                report_lines.append(f"  {i}. {warning}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        report_text = "\n".join(report_lines)
        
        # Log report
        logger.log_text(report_text, "validation_report.txt")
        print("\n" + report_text)
        
        # End MLflow run
        logger.end_run(status="FINISHED" if validation_passed else "FAILED")
        
        # Push validation results to XCom
        context['ti'].xcom_push(key='validation_passed', value=validation_passed)
        context['ti'].xcom_push(key='validation_errors', value=validation_errors)
        context['ti'].xcom_push(key='mlflow_run_id', value=logger.get_run_id())
        
        # Raise exception if validation failed
        if not validation_passed:
            raise ValueError(
                f"Data validation FAILED with {len(validation_errors)} errors. "
                f"Total nulls: {total_null_count}, Non-numeric features: {total_non_numeric_features}. "
                f"See MLflow run {logger.get_run_id()} for details."
            )
        
        return f"Validation PASSED: {files_validated} files validated successfully (MLflow Run: {logger.get_run_id()})"
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        print(f"Error during validation: {e}")
        logger.end_run(status="FAILED")
        raise


def preprocessed_eda(**context):
    """
    Perform EDA on preprocessed data and log to MLflow.
    Includes correlation heatmap and feature statistics.
    """
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
    features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")
    
    # Load transformed training data
    if metadata['split_type'] == 'simple':
        train_path = os.path.join(features_path, "train_transformed.csv")
    else:
        # Use first fold for EDA
        train_path = os.path.join(features_path, "train_fold_0_transformed.csv")
    
    print(f"Performing EDA on preprocessed data: {train_path}")
    df = pd.read_csv(train_path)
    
    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Get pipeline tag for traceability
    invocation_id = metadata.get('invocation_id', 'unknown')

    ti = context['ti']
    ti.xcom_push(key='invocation_id', value=invocation_id)

    dag_run_id = context.get('dag_run').run_id

    # Start MLflow run
    run_name = "D2S3_Preprocess_EDA"
    tags = {
        "task_type": "eda",
        "data_source": "preprocessed",
        "split_type": metadata['split_type'],
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date')),
        "airflow_dag_run_id": dag_run_id,
        "invocation_id": invocation_id,
        "pipeline_step": "D2S3"
    }
    
    try:
        logger.start_run(run_name=run_name, tags=tags)
        
        # Log preprocessed dataset version
        label_cols_str = ",".join(config.MODEL.get("LABEL_COLUMNS", ["target"]))
        logger.log_dataset(
            df=df,
            source=train_path,
            name="preprocessed_train_data",
            context="preprocessed",
            targets=label_cols_str
        )
        
        # Log dataset information
        logger.log_dataset_info(df, dataset_name="preprocessed_train")
        
        # Separate features and labels
        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in df.columns if col not in label_cols]
        
        # Get numeric and categorical columns
        numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate metrics
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "feature_columns": len(feature_cols),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "total_missing_values": int(df.isna().sum().sum()),
            "missing_percentage": round((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        }
        
        logger.log_metrics(metrics)
        logger.log_params({
            "features": ",".join(feature_cols[:50]),  # Limit to first 50 to avoid too long param
            "numeric_features": ",".join(numeric_cols[:50]),
            "categorical_features": ",".join(categorical_cols[:50])
        })
        
        # Create correlation heatmap for numeric features including target
        # Include target column(s) to show feature-target correlations
        heatmap_cols = numeric_cols.copy()
        for label_col in label_cols:
            if label_col in df.columns and label_col not in heatmap_cols:
                # Only include numeric target columns
                if pd.api.types.is_numeric_dtype(df[label_col]):
                    heatmap_cols.append(label_col)

        if len(heatmap_cols) > 1:
            print(f"Creating correlation heatmap for {len(heatmap_cols)} columns (including target)...")

            # Calculate correlation matrix
            corr_matrix = df[heatmap_cols].corr()

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=len(heatmap_cols) <= 20,  # Only annotate if <= 20 features
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )

            ax.set_title('Feature Correlation Heatmap (Preprocessed Data, incl. Target)', fontsize=14, pad=20)
            fig.tight_layout()

            # Log heatmap directly to MLflow using log_figure
            mlflow.log_figure(fig, "visualizations/correlation_heatmap.png")
            print(f"Correlation heatmap logged to MLflow: visualizations/correlation_heatmap.png")

            # Close figure to free memory
            plt.close(fig)
            
            # Log correlation statistics
            corr_flat = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            logger.log_metrics({
                "correlation_mean": float(np.mean(np.abs(corr_flat))),
                "correlation_max": float(np.max(np.abs(corr_flat))),
                "correlation_min": float(np.min(np.abs(corr_flat))),
                "high_correlation_pairs": int(np.sum(np.abs(corr_flat) > 0.8))
            })
            
            # Save correlation matrix as CSV
            corr_csv_path = os.path.join(features_path, "correlation_matrix.csv")
            corr_matrix.to_csv(corr_csv_path)
            logger.log_artifact(corr_csv_path, "correlations")
        
        # Create summary text report
        summary_lines = [
            "=" * 60,
            "PREPROCESSED DATA EDA SUMMARY",
            "=" * 60,
            f"Dataset: {train_path}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Split Type: {metadata['split_type']}",
            "",
            "DATASET OVERVIEW:",
            f"  - Total Rows: {len(df):,}",
            f"  - Total Columns: {len(df.columns)}",
            f"  - Feature Columns: {len(feature_cols)}",
            f"  - Numeric Features: {len(numeric_cols)}",
            f"  - Categorical Features: {len(categorical_cols)}",
            "",
            "DATA QUALITY:",
            f"  - Total Missing Values: {int(df.isna().sum().sum()):,}",
            f"  - Missing Percentage: {metrics['missing_percentage']}%",
            "",
            "FEATURE INFORMATION:"
        ]
        
        for col in feature_cols[:100]:  # Limit to first 100 features
            non_null = int(df[col].count())
            dtype = str(df[col].dtype)
            if col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                summary_lines.append(f"  {col}: {dtype} | Non-null: {non_null:,} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")
            else:
                unique_val = df[col].nunique()
                summary_lines.append(f"  {col}: {dtype} | Non-null: {non_null:,} | Unique: {unique_val}")
        
        if len(feature_cols) > 100:
            summary_lines.append(f"  ... and {len(feature_cols) - 100} more features")
        
        summary_lines.append("=" * 60)
        summary_text = "\n".join(summary_lines)
        
        # Log summary
        logger.log_text(summary_text, "eda_preprocessed_summary.txt")
        print("\n" + summary_text)
        
        # End MLflow run
        logger.end_run(status="FINISHED")
        
        # Push EDA summary to XCom
        context['ti'].xcom_push(key='eda_summary', value={
            'run_id': logger.get_run_id(),
            'total_rows': metrics['total_rows'],
            'feature_columns': len(feature_cols),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols)
        })
        
        return f"Preprocessed EDA completed and logged to MLflow (Run ID: {logger.get_run_id()})"
        
    except Exception as e:
        print(f"Error during preprocessed EDA: {e}")
        logger.end_run(status="FAILED")
        raise


# DAG definition
with DAG(
    dag_id="02_dag_preprocess",
    default_args=config.DEFAULT_DAG_ARGS,
    description="Preprocessing pipeline DAG - builds global preprocessing pipeline, transforms split/fold data in parallel, and performs EDA with correlation analysis on preprocessed features",
    schedule=[TRAIN_TEST_SPLIT_ASSET],  # Trigger when 01_dag_data completes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["preprocessing", "ml-pipeline"],
) as dag:
    
    # Task 1: Load split metadata from DAG 1
    metadata_load_task = PythonOperator(
        task_id="metadata_load",
        python_callable=metadata_load,
    )
    
    # Task 2: Build and save preprocessing pipeline
    pipeline_build_task = PythonOperator(
        task_id="pipeline_build",
        python_callable=pipeline_build,
        outlets=[PREPROCESSING_PIPELINE_ASSET],
    )
    
    # Task 3: Prepare transform tasks (prepares list for parallel processing)
    transform_prepare_task = transform_prepare()
    
    # Task 4: Transform splits in parallel using dynamic task mapping
    split_transform_tasks = split_transform.expand(
        split_info=transform_prepare_task
    )
    split_transform_tasks.operator.outlets = [TRANSFORMED_DATA_ASSET]  # This task produces the transformed data
    
    # Task 5: Validate transformed data
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_transformed_data,
    )
    
    # Task 6: Perform EDA on preprocessed data
    preprocessed_eda_task = PythonOperator(
        task_id="preprocessed_eda",
        python_callable=preprocessed_eda,
    )
    
    # Task dependencies
    metadata_load_task >> pipeline_build_task >> transform_prepare_task >> split_transform_tasks >> validate_data_task >> preprocessed_eda_task
