"""
Base Data DAG Template Factory.

Provides create_data_dag() factory function that creates a data pipeline DAG
with configurable experiment name and config path.
"""
from datetime import datetime
import uuid
import os
import sys
from pathlib import Path
from typing import Optional  # Used in function signatures

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.python import PythonOperator

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    raw_asset = Dataset(f"file://experiments/{experiment_name}/raw_data.csv")
    split_asset = Dataset(f"file://experiments/{experiment_name}/train_test_split.csv")
    return raw_asset, split_asset


def _load_raw_data(config, experiment_name: Optional[str] = None):
    """Create the load_raw_data task function with injected config."""
    def load_raw_data(**context):
        # Create invocation_id with timestamp + 8-char UUID suffix
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        invocation_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}Generated Pipeline Tag: {invocation_id}")

        raw_path = config.DATA.get("RAW_PATH_FILE")

        print(f"{exp_prefix}Loading data from: {raw_path}")
        df = pd.read_csv(raw_path)

        print(f"{exp_prefix}Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")

        # Push metadata to XCom
        context['ti'].xcom_push(key='invocation_id', value=invocation_id)
        context['ti'].xcom_push(key='experiment_name', value=experiment_name)
        context['ti'].xcom_push(key='data_shape', value=df.shape)
        context['ti'].xcom_push(key='data_columns', value=df.columns.tolist())
        context['ti'].xcom_push(key='raw_data_path', value=raw_path)

        return f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns (Pipeline Tag: {invocation_id})"

    return load_raw_data


def _perform_eda(config, experiment_name: Optional[str] = None):
    """Create the perform_eda task function with injected config."""
    def perform_eda(**context):
        raw_path = config.DATA.get("RAW_PATH_FILE")
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}Performing EDA on data from: {raw_path}")

        df = pd.read_csv(raw_path)

        # Initialize MLflow logger
        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        ti = context['ti']
        invocation_id = ti.xcom_pull(task_ids='raw_data_load', key='invocation_id')
        dag_run_id = context.get('dag_run').run_id

        run_name = "D1S1_Data_EDA"
        tags = {
            "task_type": "eda",
            "data_source": "raw",
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "execution_date": str(context.get('execution_date')),
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id,
            "pipeline_step": "D1S1"
        }

        if experiment_name:
            tags["experiment_name"] = experiment_name

        try:
            logger.start_run(run_name=run_name, tags=tags)

            # Log dataset version
            logger.log_dataset(
                df=df,
                source=raw_path,
                name="raw_data",
                context="raw",
                targets=",".join(config.MODEL.get("LABEL_COLUMNS", ["target"]))
            )

            # Log dataset info
            dataset_info = logger.log_dataset_info(df, dataset_name="raw_data")

            # Calculate statistics
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            metrics = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "total_missing_values": int(df.isna().sum().sum()),
                "missing_percentage": round((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                "duplicate_rows": int(df.duplicated().sum())
            }

            logger.log_metrics(metrics)

            if len(numeric_cols) > 0:
                logger.log_params({
                    "numeric_columns": ",".join(numeric_cols),
                    "categorical_columns": ",".join(categorical_cols)
                })

            # Create summary
            summary_lines = [
                "=" * 60,
                f"EXPLORATORY DATA ANALYSIS SUMMARY {f'({experiment_name})' if experiment_name else ''}",
                "=" * 60,
                f"Dataset: {raw_path}",
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "DATASET OVERVIEW:",
                f"  - Total Rows: {len(df):,}",
                f"  - Total Columns: {len(df.columns)}",
                f"  - Numeric Columns: {len(numeric_cols)}",
                f"  - Categorical Columns: {len(categorical_cols)}",
                "",
                "DATA QUALITY:",
                f"  - Total Missing Values: {int(df.isna().sum().sum()):,}",
                f"  - Missing Percentage: {metrics['missing_percentage']}%",
                f"  - Duplicate Rows: {metrics['duplicate_rows']:,}",
                "",
                "COLUMN INFORMATION:"
            ]

            for col in df.columns:
                non_null = int(df[col].count())
                null_count = int(df[col].isna().sum())
                dtype = str(df[col].dtype)
                summary_lines.append(f"  - {col}: {dtype} | Non-null: {non_null:,} | Null: {null_count:,}")

            summary_lines.append("=" * 60)
            summary_text = "\n".join(summary_lines)

            logger.log_text(summary_text, "eda_summary.txt")
            print("\n" + summary_text)

            logger.end_run(status="FINISHED")

            context['ti'].xcom_push(key='eda_summary', value={
                'run_id': logger.get_run_id(),
                'total_rows': metrics['total_rows'],
                'total_columns': metrics['total_columns'],
                'missing_percentage': metrics['missing_percentage'],
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols)
            })

            return f"EDA completed and logged to MLflow (Run ID: {logger.get_run_id()})"

        except Exception as e:
            print(f"Error during EDA: {e}")
            logger.end_run(status="FAILED")
            raise

    return perform_eda


def _split_data(config, experiment_name: Optional[str] = None):
    """Create the split_data task function with injected config."""
    def split_data(**context):
        import glob

        ti = context['ti']
        invocation_id = ti.xcom_pull(task_ids='raw_data_load', key='invocation_id')
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        raw_path = config.DATA.get("RAW_PATH_FILE")
        df = pd.read_csv(raw_path)

        fold_type = config.DATA.get("FOLD_TYPE")
        test_size = config.DATA.get("TRAIN_TEST_SPLIT", 0.2)
        num_folds = config.DATA.get("NUM_FOLDS", 5)
        shuffle = config.DATA.get("KFOLD_SHUFFLE", True)
        random_state = config.DATA.get("KFOLD_RANDOM_STATE", 42)
        stratify_col = config.DATA.get("STRATIFY_COLUMN")
        time_col = config.DATA.get("TIME_COLUMN")
        ts_gap = config.DATA.get("TIME_SERIES_GAP", 0)
        ts_expanding = config.DATA.get("TIME_SERIES_EXPANDING", True)
        processed_path = config.DATA.get("PROCESSED_PATH")

        os.makedirs(processed_path, exist_ok=True)

        print(f"{exp_prefix}Split configuration: FOLD_TYPE={fold_type}, test_size={test_size}, num_folds={num_folds}")

        # Clean up old files
        current_fold_type = fold_type if fold_type else "simple"

        if current_fold_type != "simple":
            for old_file in ["train.csv", "test.csv"]:
                old_path = os.path.join(processed_path, old_file)
                if os.path.exists(old_path):
                    os.remove(old_path)

        if current_fold_type == "simple":
            old_fold_files = glob.glob(os.path.join(processed_path, "train_fold_*.csv"))
            old_fold_files += glob.glob(os.path.join(processed_path, "val_fold_*.csv"))
            for old_file in old_fold_files:
                os.remove(old_file)

        # Sort by time column if specified
        if time_col and time_col in df.columns:
            df = df.sort_values(by=time_col).reset_index(drop=True)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in df.columns if col not in label_cols]

        X = df[feature_cols]
        y = df[label_cols] if len(label_cols) > 0 else None

        if fold_type is None or fold_type == "simple":
            print(f"{exp_prefix}Performing simple train/test split with test_size={test_size}")

            stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )

            train_df = pd.concat([X_train, y_train], axis=1) if y_train is not None else X_train
            test_df = pd.concat([X_test, y_test], axis=1) if y_test is not None else X_test

            train_path = os.path.join(processed_path, "train.csv")
            test_path = os.path.join(processed_path, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"{exp_prefix}Train set: {train_df.shape} saved to {train_path}")
            print(f"{exp_prefix}Test set: {test_df.shape} saved to {test_path}")

            # Log to MLflow
            mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
            mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

            logger = MLFlowLogger(tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name)
            dag_run_id = context.get('dag_run').run_id

            run_name = "D1S2_Data_Split"
            tags = {
                "task_type": "data_split",
                "split_type": "simple",
                "dag_id": context.get('dag').dag_id,
                "airflow_dag_run_id": dag_run_id,
                "invocation_id": invocation_id,
                "pipeline_step": "D1S2"
            }
            if experiment_name:
                tags["experiment_name"] = experiment_name

            try:
                logger.start_run(run_name=run_name, tags=tags)
                logger.log_dataset(train_df, source=train_path, name="train_split", context="training", targets=",".join(label_cols))
                logger.log_dataset(test_df, source=test_path, name="test_split", context="test", targets=",".join(label_cols))
                logger.log_params({"split_type": "simple", "test_size": test_size, "stratify_column": str(stratify_col)})
                logger.log_metrics({"train_rows": len(train_df), "test_rows": len(test_df)})
                logger.end_run(status="FINISHED")
            except Exception as e:
                print(f"Warning: Failed to log datasets to MLflow: {e}")
                if logger.get_run_id():
                    logger.end_run(status="FAILED")

            context['ti'].xcom_push(key='invocation_id', value=invocation_id)
            context['ti'].xcom_push(key='split_type', value='simple')
            context['ti'].xcom_push(key='train_shape', value=train_df.shape)
            context['ti'].xcom_push(key='test_shape', value=test_df.shape)
            context['ti'].xcom_push(key='train_path', value=train_path)
            context['ti'].xcom_push(key='test_path', value=test_path)

            return f"Simple split complete: train={train_df.shape}, test={test_df.shape}"

        elif fold_type in ("kfold", "stratified", "timeseries"):
            # Handle fold-based splits
            if fold_type == "kfold":
                kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
                splitter = kf.split(X)
            elif fold_type == "stratified":
                if stratify_col is None or stratify_col not in df.columns:
                    raise ValueError("STRATIFY_COLUMN must be specified for stratified K-Fold")
                skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
                splitter = skf.split(X, df[stratify_col])
            else:  # timeseries
                print(f"{exp_prefix}Performing Time Series Split with {num_folds} folds")
                print(f"  - Gap: {ts_gap} samples")
                print(f"  - Expanding window: {ts_expanding}")

                if ts_expanding:
                    # Use sklearn TimeSeriesSplit with gap support (expanding window)
                    tscv = TimeSeriesSplit(n_splits=num_folds, gap=ts_gap)
                    splitter = list(tscv.split(X))
                else:
                    # Sliding window: custom implementation with fixed train size
                    n_samples = len(X)
                    # Calculate window size to allow num_folds splits
                    total_available = n_samples - ts_gap
                    window_size = total_available // (num_folds + 1)  # +1 to ensure room for validation

                    if window_size < 1:
                        raise ValueError(f"Not enough data for {num_folds} folds with gap={ts_gap}. Need more samples.")

                    print(f"  - Window size: {window_size} samples")

                    splitter = []
                    for fold_idx in range(num_folds):
                        # Sliding window logic
                        train_start = fold_idx * window_size
                        train_end = train_start + window_size
                        val_start = train_end + ts_gap
                        val_end = val_start + window_size

                        if val_end > n_samples:
                            print(f"Skipping fold {fold_idx}: not enough data for validation set")
                            break

                        train_indices = list(range(train_start, train_end))
                        val_indices = list(range(val_start, val_end))
                        splitter.append((train_indices, val_indices))
                        print(f"  Fold {fold_idx} (sliding): train[{train_start}:{train_end}], gap[{train_end}:{val_start}], val[{val_start}:{val_end}]")

            fold_paths = []
            for fold_idx, (train_idx, val_idx) in enumerate(splitter):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx] if y is not None else None
                y_val_fold = y.iloc[val_idx] if y is not None else None

                train_fold_df = pd.concat([X_train_fold, y_train_fold], axis=1) if y_train_fold is not None else X_train_fold
                val_fold_df = pd.concat([X_val_fold, y_val_fold], axis=1) if y_val_fold is not None else X_val_fold

                train_fold_path = os.path.join(processed_path, f"train_fold_{fold_idx}.csv")
                val_fold_path = os.path.join(processed_path, f"val_fold_{fold_idx}.csv")

                train_fold_df.to_csv(train_fold_path, index=False)
                val_fold_df.to_csv(val_fold_path, index=False)

                fold_paths.append({"train": train_fold_path, "val": val_fold_path})
                print(f"{exp_prefix}Fold {fold_idx}: train={train_fold_df.shape}, val={val_fold_df.shape}")

            context['ti'].xcom_push(key='invocation_id', value=invocation_id)
            context['ti'].xcom_push(key='split_type', value=fold_type)
            context['ti'].xcom_push(key='num_folds', value=len(fold_paths))
            context['ti'].xcom_push(key='fold_paths', value=fold_paths)

            # Add time series specific metadata
            if fold_type == "timeseries":
                context['ti'].xcom_push(key='ts_gap', value=ts_gap)
                context['ti'].xcom_push(key='ts_expanding', value=ts_expanding)

            return f"{fold_type} split complete: {len(fold_paths)} folds created{f' (gap={ts_gap}, expanding={ts_expanding})' if fold_type == 'timeseries' else ''}"

        else:
            raise ValueError(f"Unknown FOLD_TYPE: {fold_type}")

    return split_data


def create_data_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a data pipeline DAG.

    Args:
        experiment_name: Optional experiment name for scoping.
                        If None, creates default "01_dag_data" DAG.
        config_path: Optional path to experiment-specific config.json.
                    If None, uses default config location.

    Returns:
        Configured Airflow DAG object.

    Examples:
        # Default DAG (backward compatible)
        dag = create_data_dag()

        # Experiment-specific DAG
        dag = create_data_dag(
            experiment_name="housing",
            config_path="/opt/airflow/dags/experiments/housing/config.json"
        )
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    raw_asset, split_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_01_dag_data"
    description = f"Data pipeline for {experiment_name} experiment"
    tags = ["data", "ml-pipeline", "experiment", experiment_name]

    # Create DAG
    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        is_paused_upon_creation=False,
        tags=tags,
    )

    with dag:
        # Task 1: Load raw data
        raw_data_load = PythonOperator(
            task_id="raw_data_load",
            python_callable=_load_raw_data(config, experiment_name),
            outlets=[raw_asset],
        )

        # Task 2: Perform EDA
        eda_task = PythonOperator(
            task_id="eda_task",
            python_callable=_perform_eda(config, experiment_name),
        )

        # Task 3: Split data
        data_split = PythonOperator(
            task_id="data_split",
            python_callable=_split_data(config, experiment_name),
            outlets=[split_asset],
        )

        raw_data_load >> eda_task >> data_split

    return dag
