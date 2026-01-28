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
    Load split metadata from 01_dag_data XCom.
    Returns information about split type and file paths.
    """
    ti = context['ti']
    
    # Pull metadata from previous DAG run (01_dag_data)
    # include_prior_dates=True returns a list, so we take the first (most recent) value
    split_type_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='split_type', include_prior_dates=True)
    
    if split_type_list is None or len(split_type_list) == 0:
        raise ValueError("No split metadata found from 01_dag_data. Ensure 01_dag_data has run successfully.")
    
    split_type = split_type_list[0] if isinstance(split_type_list, list) else split_type_list
    
    print(f"Split type detected: {split_type}")
    
    metadata = {
        'split_type': split_type
    }
    
    if split_type == 'simple':
        train_path_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='train_path', include_prior_dates=True)
        test_path_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='test_path', include_prior_dates=True)
        train_shape_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='train_shape', include_prior_dates=True)
        test_shape_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='test_shape', include_prior_dates=True)
        
        metadata['train_path'] = train_path_list[0] if isinstance(train_path_list, list) else train_path_list
        metadata['test_path'] = test_path_list[0] if isinstance(test_path_list, list) else test_path_list
        metadata['train_shape'] = train_shape_list[0] if isinstance(train_shape_list, list) else train_shape_list
        metadata['test_shape'] = test_shape_list[0] if isinstance(test_shape_list, list) else test_shape_list
        print(f"Simple split: train={metadata['train_shape']}, test={metadata['test_shape']}")
        
    elif split_type in ['kfold', 'stratified', 'timeseries']:
        num_folds_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='num_folds', include_prior_dates=True)
        fold_paths_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='fold_paths', include_prior_dates=True)
        
        metadata['num_folds'] = num_folds_list[0] if isinstance(num_folds_list, list) else num_folds_list
        metadata['fold_paths'] = fold_paths_list[0] if isinstance(fold_paths_list, list) else fold_paths_list
        print(f"{split_type} split: {metadata['num_folds']} folds")
        
        if split_type == 'timeseries':
            ts_gap_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='ts_gap', include_prior_dates=True)
            ts_expanding_list = ti.xcom_pull(dag_id='01_dag_data', task_ids='data_split', key='ts_expanding', include_prior_dates=True)
            metadata['ts_gap'] = ts_gap_list[0] if isinstance(ts_gap_list, list) else ts_gap_list
            metadata['ts_expanding'] = ts_expanding_list[0] if isinstance(ts_expanding_list, list) else ts_expanding_list
    
    # Push metadata for downstream tasks
    context['ti'].xcom_push(key='split_metadata', value=metadata)
    
    return f"Loaded {split_type} split metadata"


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
    mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Start MLflow run
    dag_run_id = context.get('dag_run').run_id
    run_name = f"{dag_run_id}_Pipeline_Build"
    tags = {
        "task_type": "preprocessing_pipeline",
        "split_type": metadata['split_type'],
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date')),
        "airflow_dag_run_id": dag_run_id
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
        
        # End MLflow run
        logger.end_run(status="FINISHED")
        
        # Push additional metadata to XCom
        context['ti'].xcom_push(key='mlflow_run_id', value=logger.get_run_id())
        
        print(f"Pipeline logged to MLflow (Run ID: {logger.get_run_id()})")
        
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
    mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Start MLflow run
    dag_run_id = context.get('dag_run').run_id
    run_name = f"{dag_run_id}_EDA_Preprocessed"
    tags = {
        "task_type": "eda",
        "data_source": "preprocessed",
        "split_type": metadata['split_type'],
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date')),
        "airflow_dag_run_id": dag_run_id
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
        
        # Create correlation heatmap for numeric features
        if len(numeric_cols) > 1:
            print(f"Creating correlation heatmap for {len(numeric_cols)} numeric features...")
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=len(numeric_cols) <= 20,  # Only annotate if <= 20 features
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            
            plt.title('Feature Correlation Heatmap (Preprocessed Data)', fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(features_path, "correlation_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log heatmap to MLflow
            logger.log_artifact(heatmap_path, "visualizations")
            print(f"Correlation heatmap saved and logged: {heatmap_path}")
            
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
    
    # Task 5: Perform EDA on preprocessed data
    preprocessed_eda_task = PythonOperator(
        task_id="preprocessed_eda",
        python_callable=preprocessed_eda,
    )
    
    # Task dependencies
    metadata_load_task >> pipeline_build_task >> transform_prepare_task >> split_transform_tasks >> preprocessed_eda_task
