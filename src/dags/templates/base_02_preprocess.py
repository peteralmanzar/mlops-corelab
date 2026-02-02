"""
Base Preprocessing DAG Template Factory.

Provides create_preprocess_dag() factory function that creates a preprocessing
pipeline DAG with configurable experiment name and config path.

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
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.python import PythonOperator
from airflow.decorators import task

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger
from data_pipeline import build_pipeline, save_pipeline, load_pipeline


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    split_asset = Dataset(f"file://experiments/{experiment_name}/train_test_split.csv")
    pipeline_asset = Dataset(f"file://experiments/{experiment_name}/preprocessing_pipeline.joblib")
    transformed_asset = Dataset(f"file://experiments/{experiment_name}/transformed_data.csv")
    return split_asset, pipeline_asset, transformed_asset


def _metadata_load(config, experiment_name: Optional[str] = None):
    """Create the metadata_load task function with injected config."""
    def metadata_load(**context):
        """
        Load split metadata by reading config directly and detecting available files.
        """
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        # Read split type directly from config (source of truth)
        fold_type = config.DATA.get("FOLD_TYPE")
        split_type = fold_type if fold_type else "simple"

        print(f"{exp_prefix}Split type from config: {split_type}")

        # Verify files exist for this split type
        processed_path = config.DATA.get("PROCESSED_PATH")

        if split_type == "simple":
            train_file = os.path.join(processed_path, "train.csv")
            test_file = os.path.join(processed_path, "test.csv")
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                raise FileNotFoundError(
                    f"Simple split files not found in {processed_path}. "
                    f"Expected: train.csv, test.csv. "
                    f"Please run 01_dag_data with FOLD_TYPE=null/simple in config."
                )
        else:
            import glob
            fold_train_files = glob.glob(os.path.join(processed_path, "train_fold_*.csv"))
            if not fold_train_files:
                raise FileNotFoundError(
                    f"No fold files found in {processed_path}. "
                    f"Expected: train_fold_*.csv files. "
                    f"Config shows FOLD_TYPE={fold_type}. "
                    f"Please run 01_dag_data to generate {fold_type} splits."
                )
            print(f"{exp_prefix}Found {len(fold_train_files)} fold files")

        print(f"{exp_prefix}Split type verified: {split_type}")

        # Build metadata from actual files on disk
        metadata = {
            'split_type': split_type,
            'experiment_name': experiment_name
        }

        if split_type == 'simple':
            train_path = os.path.join(processed_path, "train.csv")
            test_path = os.path.join(processed_path, "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            metadata['train_path'] = train_path
            metadata['test_path'] = test_path
            metadata['train_shape'] = train_df.shape
            metadata['test_shape'] = test_df.shape
            print(f"{exp_prefix}Simple split: train={metadata['train_shape']}, test={metadata['test_shape']}")

        elif split_type in ['kfold', 'stratified', 'timeseries']:
            import glob

            train_files = sorted(glob.glob(os.path.join(processed_path, "train_fold_*.csv")))
            val_files = sorted(glob.glob(os.path.join(processed_path, "val_fold_*.csv")))

            if len(train_files) == 0:
                raise FileNotFoundError(f"No train_fold_*.csv files found in {processed_path}")
            if len(val_files) == 0:
                raise FileNotFoundError(f"No val_fold_*.csv files found in {processed_path}")

            num_folds = len(train_files)
            fold_paths = []

            for i in range(num_folds):
                train_fold_path = train_files[i]
                val_fold_path = val_files[i]

                fold_paths.append({
                    'fold': i,
                    'train': train_fold_path,
                    'test': val_fold_path
                })

            metadata['num_folds'] = num_folds
            metadata['fold_paths'] = fold_paths
            print(f"{exp_prefix}{split_type} split: {num_folds} folds detected")

            if split_type == 'timeseries':
                metadata['ts_gap'] = config.DATA.get("TIME_SERIES_GAP", 0)
                metadata['ts_expanding'] = config.DATA.get("TIME_SERIES_EXPANDING", True)

        # Get the triggering asset events to find the correct upstream DAG run
        triggering_events = context.get('triggering_asset_events')
        upstream_run_id = None

        if triggering_events:
            for asset_uri, events in triggering_events.items():
                for event in events:
                    if hasattr(event, 'source_run_id'):
                        upstream_run_id = event.source_run_id
                        break
                    elif hasattr(event, 'extra') and event.extra:
                        upstream_run_id = event.extra.get('run_id')
                        break
                if upstream_run_id:
                    break

        # Pull invocation_id using specific run_id if available
        upstream_dag_id = f"{experiment_name}_01_dag_data" if experiment_name else "01_dag_data"
        invocation_id = None
        if upstream_run_id:
            invocation_id = ti.xcom_pull(dag_id=upstream_dag_id, task_ids='data_split', key='invocation_id', run_id=upstream_run_id)

        if not invocation_id:
            invocation_id_list = ti.xcom_pull(dag_id=upstream_dag_id, task_ids='data_split', key='invocation_id', include_prior_dates=True)
            if invocation_id_list is not None:
                if isinstance(invocation_id_list, list) and len(invocation_id_list) > 0:
                    invocation_id = invocation_id_list[0]
                elif isinstance(invocation_id_list, str):
                    invocation_id = invocation_id_list

        if invocation_id:
            print(f"{exp_prefix}Pipeline Tag: {invocation_id}")
            metadata['invocation_id'] = invocation_id
        else:
            raise ValueError(f"invocation_id not found from {upstream_dag_id}. Ensure 01_dag_data completed successfully.")

        # Push metadata for downstream tasks
        context['ti'].xcom_push(key='split_metadata', value=metadata)
        context['ti'].xcom_push(key='invocation_id', value=invocation_id)

        return f"Loaded {split_type} split metadata (Pipeline Tag: {invocation_id})"

    return metadata_load


def _pipeline_build(config, experiment_name: Optional[str] = None):
    """Create the pipeline_build task function with injected config."""
    def pipeline_build(**context):
        """
        Build preprocessing pipeline from config and save it.
        """
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}Building preprocessing pipeline from config...")

        pipeline = build_pipeline(config)

        artifacts_path = config.MODEL.get("ARTIFACTS_PATH", "/mlflow/artifacts")
        os.makedirs(artifacts_path, exist_ok=True)
        pipeline_path = os.path.join(artifacts_path, "preprocessing_pipeline.joblib")

        ti = context['ti']
        metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')

        if metadata['split_type'] == 'simple':
            train_path = metadata['train_path']
        else:
            train_path = metadata['fold_paths'][0]['train']

        print(f"{exp_prefix}Fitting pipeline on: {train_path}")
        train_df = pd.read_csv(train_path)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in train_df.columns if col not in label_cols]
        X_train = train_df[feature_cols]

        print(f"{exp_prefix}Original features shape: {X_train.shape}")
        pipeline.fit(X_train)

        X_transformed = pipeline.transform(X_train)
        print(f"{exp_prefix}Transformed features shape: {X_transformed.shape}")

        save_pipeline(pipeline, pipeline_path)
        print(f"{exp_prefix}Pipeline saved to: {pipeline_path}")

        # Initialize MLflow logger and log pipeline
        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        invocation_id = metadata.get('invocation_id', 'unknown')
        dag_run_id = context.get('dag_run').run_id

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

        if experiment_name:
            tags["experiment_name"] = experiment_name

        try:
            logger.start_run(run_name=run_name, tags=tags)

            logger.log_dataset(
                df=train_df,
                source=train_path,
                name="pipeline_training_data",
                context="pipeline_fit",
                targets=",".join(label_cols)
            )

            pipeline_params = {
                "n_steps": len(pipeline.steps),
                "step_names": ",".join([step[0] for step in pipeline.steps]),
                "input_features": len(feature_cols),
                "original_shape": str(X_train.shape),
                "transformed_shape": str(X_transformed.shape),
                "training_data_path": train_path
            }
            logger.log_params(pipeline_params)

            metrics = {
                "n_pipeline_steps": len(pipeline.steps),
                "n_input_features": X_train.shape[1],
                "n_output_features": X_transformed.shape[1] if hasattr(X_transformed, 'shape') else len(X_transformed.columns),
            }
            if X_train.shape[1] > 0:
                output_features = X_transformed.shape[1] if hasattr(X_transformed, 'shape') else len(X_transformed.columns)
                metrics["dimensionality_reduction_ratio"] = round(X_train.shape[1] / output_features, 4)
            logger.log_metrics(metrics)

            logger.log_sklearn_pipeline(
                pipeline=pipeline,
                artifact_path="preprocessing_pipeline",
                input_example=X_train.head(5) if len(X_train) >= 5 else X_train
            )

            logger.log_artifact(pipeline_path, "pipeline_artifacts")

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

            run_id = logger.get_run_id()
            context['ti'].xcom_push(key='mlflow_run_id', value=run_id)
            context['ti'].xcom_push(key='pipeline_mlflow_run_id', value=run_id)

            logger.end_run(status="FINISHED")
            print(f"{exp_prefix}Pipeline logged to MLflow (Run ID: {run_id})")

        except Exception as e:
            print(f"{exp_prefix}Error logging pipeline to MLflow: {e}")
            logger.end_run(status="FAILED")

        context['ti'].xcom_push(key='pipeline_path', value=pipeline_path)
        context['ti'].xcom_push(key='original_shape', value=X_train.shape)
        context['ti'].xcom_push(key='transformed_shape', value=X_transformed.shape)

        return f"Pipeline built and saved: {pipeline_path}"

    return pipeline_build


def _create_split_transform_task(config, experiment_name: Optional[str] = None):
    """Create the split_transform task function."""
    @task
    def split_transform(split_info: dict):
        """Transform a single train/test split using the pipeline."""
        fold = split_info.get('fold', 'simple')
        train_path = split_info['train']
        test_path = split_info['test']
        pipeline_path = split_info['pipeline_path']
        features_path = split_info['features_path']
        label_cols = split_info['label_cols']

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}Processing fold: {fold}")

        pipeline = load_pipeline(pipeline_path)

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df[[col for col in train_df.columns if col not in label_cols]]
        y_train = train_df[label_cols] if label_cols else None

        X_test = test_df[[col for col in test_df.columns if col not in label_cols]]
        y_test = test_df[label_cols] if label_cols else None

        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        if isinstance(X_train_transformed, np.ndarray):
            X_train_transformed = pd.DataFrame(X_train_transformed)
            X_test_transformed = pd.DataFrame(X_test_transformed)

        if y_train is not None:
            train_transformed = pd.concat([X_train_transformed, y_train.reset_index(drop=True)], axis=1)
        else:
            train_transformed = X_train_transformed

        if y_test is not None:
            test_transformed = pd.concat([X_test_transformed, y_test.reset_index(drop=True)], axis=1)
        else:
            test_transformed = X_test_transformed

        os.makedirs(features_path, exist_ok=True)

        if fold == 'simple':
            train_output = os.path.join(features_path, "train_transformed.csv")
            test_output = os.path.join(features_path, "test_transformed.csv")
        else:
            train_output = os.path.join(features_path, f"train_fold_{fold}_transformed.csv")
            test_output = os.path.join(features_path, f"test_fold_{fold}_transformed.csv")

        train_transformed.to_csv(train_output, index=False)
        test_transformed.to_csv(test_output, index=False)

        print(f"{exp_prefix}  Saved: {train_output} (shape: {train_transformed.shape})")
        print(f"{exp_prefix}  Saved: {test_output} (shape: {test_transformed.shape})")

        return {
            'fold': fold,
            'train_output': train_output,
            'test_output': test_output,
            'train_shape': train_transformed.shape,
            'test_shape': test_transformed.shape
        }

    return split_transform


def _create_transform_prepare_task(config, experiment_name: Optional[str] = None):
    """Create the transform_prepare task function."""
    @task
    def transform_prepare(**context):
        """Prepare the list of splits to transform in parallel."""
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

    return transform_prepare


def _validate_transformed_data(config, experiment_name: Optional[str] = None):
    """Create the validate_transformed_data task function."""
    def validate_transformed_data(**context):
        """Validate transformed data for strict quality requirements."""
        ti = context['ti']
        metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
        features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")
        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}Starting data validation...")

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

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        invocation_id = metadata.get('invocation_id', 'unknown')
        dag_run_id = context.get('dag_run').run_id

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

        if experiment_name:
            tags["experiment_name"] = experiment_name

        validation_passed = True
        validation_errors = []
        total_null_count = 0
        total_non_numeric_features = 0
        files_validated = 0

        try:
            logger.start_run(run_name=run_name, tags=tags)

            for file_info in files_to_validate:
                file_name = file_info['name']
                file_path = file_info['path']

                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    validation_errors.append(error_msg)
                    print(f"{exp_prefix}ERROR: {error_msg}")
                    validation_passed = False
                    continue

                print(f"{exp_prefix}Validating: {file_name}")
                df = pd.read_csv(file_path)

                feature_cols = [col for col in df.columns if col not in label_cols]

                null_count = df.isnull().sum().sum()
                if null_count > 0:
                    null_details = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
                    error_msg = f"{file_name}: Found {null_count} null values in columns: {null_details}"
                    validation_errors.append(error_msg)
                    print(f"{exp_prefix}  ❌ NULL CHECK FAILED: {null_count} null values found")
                    validation_passed = False
                    total_null_count += null_count
                else:
                    print(f"{exp_prefix}  ✓ NULL CHECK PASSED: Zero null values")

                non_numeric_features = df[feature_cols].select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                if len(non_numeric_features) > 0:
                    non_numeric_dtypes = {col: str(df[col].dtype) for col in non_numeric_features}
                    error_msg = f"{file_name}: Found {len(non_numeric_features)} non-numeric feature columns: {non_numeric_dtypes}"
                    validation_errors.append(error_msg)
                    print(f"{exp_prefix}  ❌ NUMERIC CHECK FAILED: {len(non_numeric_features)} non-numeric features")
                    validation_passed = False
                    total_non_numeric_features += len(non_numeric_features)
                else:
                    print(f"{exp_prefix}  ✓ NUMERIC CHECK PASSED: All {len(feature_cols)} features are numeric")

                logger.log_metrics({
                    f"{file_name}_null_count": int(null_count),
                    f"{file_name}_non_numeric_count": len(non_numeric_features),
                    f"{file_name}_total_rows": len(df),
                    f"{file_name}_total_features": len(feature_cols)
                })

                files_validated += 1

            logger.log_metrics({
                "validation_passed": 1 if validation_passed else 0,
                "total_null_count": total_null_count,
                "total_non_numeric_features": total_non_numeric_features,
                "files_validated": files_validated,
                "validation_errors": len(validation_errors)
            })

            logger.log_params({
                "validation_rule_nulls": "zero_allowed",
                "validation_rule_types": "all_numeric_features",
                "split_type": metadata['split_type'],
                "files_checked": files_validated
            })

            report_lines = [
                "=" * 60,
                f"DATA VALIDATION REPORT {f'({experiment_name})' if experiment_name else ''}",
                "=" * 60,
                f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Split Type: {metadata['split_type']}",
                f"Files Validated: {files_validated}",
                "",
                f"VALIDATION STATUS: {'✓ PASSED' if validation_passed else '✗ FAILED'}",
                "",
                f"SUMMARY:",
                f"  - Total Null Values: {total_null_count}",
                f"  - Non-Numeric Features: {total_non_numeric_features}",
                f"  - Errors: {len(validation_errors)}",
                "=" * 60
            ]

            logger.log_text("\n".join(report_lines), "validation_report.txt")
            print("\n" + "\n".join(report_lines))

            logger.end_run(status="FINISHED" if validation_passed else "FAILED")

            context['ti'].xcom_push(key='validation_passed', value=validation_passed)
            context['ti'].xcom_push(key='validation_errors', value=validation_errors)
            context['ti'].xcom_push(key='mlflow_run_id', value=logger.get_run_id())

            if not validation_passed:
                raise ValueError(
                    f"Data validation FAILED with {len(validation_errors)} errors. "
                    f"Total nulls: {total_null_count}, Non-numeric features: {total_non_numeric_features}."
                )

            return f"Validation PASSED: {files_validated} files validated successfully"

        except ValueError:
            raise
        except Exception as e:
            print(f"{exp_prefix}Error during validation: {e}")
            logger.end_run(status="FAILED")
            raise

    return validate_transformed_data


def _preprocessed_eda(config, experiment_name: Optional[str] = None):
    """Create the preprocessed_eda task function."""
    def preprocessed_eda(**context):
        """Perform EDA on preprocessed data and log to MLflow."""
        ti = context['ti']
        metadata = ti.xcom_pull(task_ids='metadata_load', key='split_metadata')
        features_path = config.PREPROCESSING.get("FEATURES_PATH", "/home/jovyan/data/features")

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        if metadata['split_type'] == 'simple':
            train_path = os.path.join(features_path, "train_transformed.csv")
        else:
            train_path = os.path.join(features_path, "train_fold_0_transformed.csv")

        print(f"{exp_prefix}Performing EDA on preprocessed data: {train_path}")
        df = pd.read_csv(train_path)

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        invocation_id = metadata.get('invocation_id', 'unknown')
        ti.xcom_push(key='invocation_id', value=invocation_id)

        dag_run_id = context.get('dag_run').run_id

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

        if experiment_name:
            tags["experiment_name"] = experiment_name

        try:
            logger.start_run(run_name=run_name, tags=tags)

            label_cols_str = ",".join(config.MODEL.get("LABEL_COLUMNS", ["target"]))
            logger.log_dataset(
                df=df,
                source=train_path,
                name="preprocessed_train_data",
                context="preprocessed",
                targets=label_cols_str
            )

            logger.log_dataset_info(df, dataset_name="preprocessed_train")

            label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
            feature_cols = [col for col in df.columns if col not in label_cols]

            numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

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
                "features": ",".join(feature_cols[:50]),
                "numeric_features": ",".join(numeric_cols[:50]),
                "categorical_features": ",".join(categorical_cols[:50])
            })

            # Create correlation heatmap
            heatmap_cols = numeric_cols.copy()
            for label_col in label_cols:
                if label_col in df.columns and label_col not in heatmap_cols:
                    if pd.api.types.is_numeric_dtype(df[label_col]):
                        heatmap_cols.append(label_col)

            if len(heatmap_cols) > 1:
                print(f"{exp_prefix}Creating correlation heatmap for {len(heatmap_cols)} columns...")

                corr_matrix = df[heatmap_cols].corr()

                fig, ax = plt.subplots(figsize=(12, 10))

                sns.heatmap(
                    corr_matrix,
                    annot=len(heatmap_cols) <= 20,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=ax
                )

                ax.set_title('Feature Correlation Heatmap (Preprocessed Data)', fontsize=14, pad=20)
                fig.tight_layout()

                mlflow.log_figure(fig, "visualizations/correlation_heatmap.png")
                plt.close(fig)

                corr_flat = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                logger.log_metrics({
                    "correlation_mean": float(np.mean(np.abs(corr_flat))),
                    "correlation_max": float(np.max(np.abs(corr_flat))),
                    "correlation_min": float(np.min(np.abs(corr_flat))),
                    "high_correlation_pairs": int(np.sum(np.abs(corr_flat) > 0.8))
                })

                corr_csv_path = os.path.join(features_path, "correlation_matrix.csv")
                corr_matrix.to_csv(corr_csv_path)
                logger.log_artifact(corr_csv_path, "correlations")

            summary_lines = [
                "=" * 60,
                f"PREPROCESSED DATA EDA SUMMARY {f'({experiment_name})' if experiment_name else ''}",
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
                "=" * 60
            ]

            logger.log_text("\n".join(summary_lines), "eda_preprocessed_summary.txt")
            print("\n" + "\n".join(summary_lines))

            logger.end_run(status="FINISHED")

            context['ti'].xcom_push(key='eda_summary', value={
                'run_id': logger.get_run_id(),
                'total_rows': metrics['total_rows'],
                'feature_columns': len(feature_cols),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols)
            })

            return f"Preprocessed EDA completed and logged to MLflow (Run ID: {logger.get_run_id()})"

        except Exception as e:
            print(f"{exp_prefix}Error during preprocessed EDA: {e}")
            logger.end_run(status="FAILED")
            raise

    return preprocessed_eda


def create_preprocess_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a preprocessing pipeline DAG.

    Args:
        experiment_name: Experiment name for scoping.
        config_path: Optional path to experiment-specific config.json.

    Returns:
        Configured Airflow DAG object.
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    split_asset, pipeline_asset, transformed_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_02_dag_preprocess"
    description = f"Preprocessing pipeline for {experiment_name} experiment"
    tags = ["preprocessing", "ml-pipeline", "experiment", experiment_name]

    # Create DAG
    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[split_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    with dag:
        # Task 1: Load split metadata from DAG 1
        metadata_load_task = PythonOperator(
            task_id="metadata_load",
            python_callable=_metadata_load(config, experiment_name),
        )

        # Task 2: Build and save preprocessing pipeline
        pipeline_build_task = PythonOperator(
            task_id="pipeline_build",
            python_callable=_pipeline_build(config, experiment_name),
            outlets=[pipeline_asset],
        )

        # Task 3: Prepare transform tasks
        transform_prepare_task = _create_transform_prepare_task(config, experiment_name)()

        # Task 4: Transform splits in parallel using dynamic task mapping
        split_transform = _create_split_transform_task(config, experiment_name)
        split_transform_tasks = split_transform.expand(
            split_info=transform_prepare_task
        )

        # Task 5: Validate transformed data
        validate_data_task = PythonOperator(
            task_id="validate_data",
            python_callable=_validate_transformed_data(config, experiment_name),
            outlets=[transformed_asset],
        )

        # Task 6: Perform EDA on preprocessed data
        preprocessed_eda_task = PythonOperator(
            task_id="preprocessed_eda",
            python_callable=_preprocessed_eda(config, experiment_name),
        )

        # Task dependencies
        metadata_load_task >> pipeline_build_task >> transform_prepare_task >> split_transform_tasks >> validate_data_task >> preprocessed_eda_task

    return dag
