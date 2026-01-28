"""
01_dag_data - Data Pipeline DAG

This DAG handles comprehensive data operations for the ML pipeline:
- Loads raw data from configured source
- Performs exploratory data analysis (EDA) with MLflow logging
- Creates train/test splits with support for multiple cross-validation strategies:
  * Simple train/test split
  * K-Fold cross-validation
  * Stratified K-Fold cross-validation
  * Time Series split with gap and expanding/sliding window support
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

# Add utils to path for config loading
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger

# Import assets for data-aware scheduling
from assets import RAW_DATA_ASSET, TRAIN_TEST_SPLIT_ASSET

# Load configuration
config = Config.load()


def load_raw_data(**context):
    """
    Load raw data from configured path into a pandas DataFrame.
    Stores the DataFrame shape in XCom for validation.
    """
    raw_path = config.DATA.get("RAW_PATH_FILE")
    
    print(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Push metadata to XCom for downstream tasks
    context['ti'].xcom_push(key='data_shape', value=df.shape)
    context['ti'].xcom_push(key='data_columns', value=df.columns.tolist())
    context['ti'].xcom_push(key='raw_data_path', value=raw_path)
    
    return f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns"


def perform_eda(**context):
    """
    Perform Exploratory Data Analysis (EDA) on raw data and log to MLflow.
    
    Logs comprehensive dataset information including:
    - Column names and data types
    - Total number of rows
    - Non-null and null counts per column
    - Basic statistics
    """
    # Load raw data
    raw_path = config.DATA.get("RAW_PATH_FILE")
    print(f"Performing EDA on data from: {raw_path}")
    
    df = pd.read_csv(raw_path)
    
    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
    
    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )
    
    # Start MLflow run
    run_name = f"EDA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tags = {
        "task_type": "eda",
        "data_source": "raw",
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "execution_date": str(context.get('execution_date'))
    }
    
    try:
        logger.start_run(run_name=run_name, tags=tags)
        
        # Log dataset version using MLflow dataset tracking
        logger.log_dataset(
            df=df,
            source=raw_path,
            name="raw_data",
            context="raw",
            targets=",".join(config.MODEL.get("LABEL_COLUMNS", ["target"]))
        )
        
        # Log dataset information (columns, dtypes, null counts, etc.)
        dataset_info = logger.log_dataset_info(df, dataset_name="raw_data")
        
        # Calculate and log additional statistics
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
        
        # Log basic statistics for numeric columns
        if len(numeric_cols) > 0:
            stats_dict = df[numeric_cols].describe().to_dict()
            logger.log_params({
                "numeric_columns": ",".join(numeric_cols),
                "categorical_columns": ",".join(categorical_cols)
            })
        
        # Create summary text report
        summary_lines = [
            "=" * 60,
            "EXPLORATORY DATA ANALYSIS SUMMARY",
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
            summary_lines.append(
                f"  - {col}: {dtype} | Non-null: {non_null:,} | Null: {null_count:,}"
            )
        
        summary_lines.append("=" * 60)
        summary_text = "\n".join(summary_lines)
        
        # Log summary as text artifact
        logger.log_text(summary_text, "eda_summary.txt")
        
        print("\n" + summary_text)
        
        # End MLflow run
        logger.end_run(status="FINISHED")
        
        # Push EDA summary to XCom
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


def split_data(**context):
    """
    Split data based on config:
    - If FOLD_TYPE is None/null: Simple train/test split
    - If FOLD_TYPE is 'kfold': K-Fold cross-validation
    - If FOLD_TYPE is 'stratified': Stratified K-Fold
    - If FOLD_TYPE is 'timeseries': Time Series Split with gap and expanding window support
    """
    # Reload data
    raw_path = config.DATA.get("RAW_PATH_FILE")
    df = pd.read_csv(raw_path)
    
    # Get config values
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
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)
    
    print(f"Split configuration: FOLD_TYPE={fold_type}, test_size={test_size}, num_folds={num_folds}")
    
    # Sort by time column if specified (for timeseries integrity)
    if time_col and time_col in df.columns:
        print(f"Sorting data by time column: {time_col}")
        df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Separate features and labels
    label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    X = df[feature_cols]
    y = df[label_cols] if len(label_cols) > 0 else None
    
    if fold_type is None or fold_type == "simple":
        # Simple train/test split
        print(f"Performing simple train/test split with test_size={test_size}")
        
        stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        # Save splits
        train_df = pd.concat([X_train, y_train], axis=1) if y_train is not None else X_train
        test_df = pd.concat([X_test, y_test], axis=1) if y_test is not None else X_test
        
        train_path = os.path.join(processed_path, "train.csv")
        test_path = os.path.join(processed_path, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train set: {train_df.shape} saved to {train_path}")
        print(f"Test set: {test_df.shape} saved to {test_path}")
        
        # Log datasets to MLflow for versioning
        mlflow_tracking_uri = config.MLFLOW.get("MLFLOW_TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("MLFLOW_EXPERIMENT_NAME")
        logger = MLFlowLogger(tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name)
        
        run_name = f"Data_Split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tags = {"task_type": "data_split", "split_type": "simple", "dag_id": context.get('dag').dag_id}
        
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
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='split_type', value='simple')
        context['ti'].xcom_push(key='train_shape', value=train_df.shape)
        context['ti'].xcom_push(key='test_shape', value=test_df.shape)
        context['ti'].xcom_push(key='train_path', value=train_path)
        context['ti'].xcom_push(key='test_path', value=test_path)
        
        return f"Simple split complete: train={train_df.shape}, test={test_df.shape}"
        
    elif fold_type == "kfold":
        # K-Fold cross-validation
        print(f"Performing K-Fold cross-validation with {num_folds} folds")
        
        kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
        
        fold_paths = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = (y.iloc[train_idx], y.iloc[val_idx]) if y is not None else (None, None)
            
            # Save fold
            train_fold_df = pd.concat([X_train_fold, y_train_fold], axis=1) if y_train_fold is not None else X_train_fold
            val_fold_df = pd.concat([X_val_fold, y_val_fold], axis=1) if y_val_fold is not None else X_val_fold
            
            train_fold_path = os.path.join(processed_path, f"train_fold_{fold_idx}.csv")
            val_fold_path = os.path.join(processed_path, f"val_fold_{fold_idx}.csv")
            
            train_fold_df.to_csv(train_fold_path, index=False)
            val_fold_df.to_csv(val_fold_path, index=False)
            
            fold_paths.append({"train": train_fold_path, "val": val_fold_path})
            print(f"Fold {fold_idx}: train={train_fold_df.shape}, val={val_fold_df.shape}")
        
        # Push metadata to XCom
        context['ti'].xcom_push(key='split_type', value='kfold')
        context['ti'].xcom_push(key='num_folds', value=num_folds)
        context['ti'].xcom_push(key='fold_paths', value=fold_paths)
        
        return f"K-Fold split complete: {num_folds} folds created"
        
    elif fold_type == "stratified":
        # Stratified K-Fold
        if stratify_col is None or stratify_col not in df.columns:
            raise ValueError(f"STRATIFY_COLUMN must be specified for stratified K-Fold")
        
        print(f"Performing Stratified K-Fold with {num_folds} folds on column '{stratify_col}'")
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
        
        fold_paths = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, df[stratify_col])):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = (y.iloc[train_idx], y.iloc[val_idx]) if y is not None else (None, None)
            
            train_fold_df = pd.concat([X_train_fold, y_train_fold], axis=1) if y_train_fold is not None else X_train_fold
            val_fold_df = pd.concat([X_val_fold, y_val_fold], axis=1) if y_val_fold is not None else X_val_fold
            
            train_fold_path = os.path.join(processed_path, f"train_fold_{fold_idx}.csv")
            val_fold_path = os.path.join(processed_path, f"val_fold_{fold_idx}.csv")
            
            train_fold_df.to_csv(train_fold_path, index=False)
            val_fold_df.to_csv(val_fold_path, index=False)
            
            fold_paths.append({"train": train_fold_path, "val": val_fold_path})
            print(f"Fold {fold_idx}: train={train_fold_df.shape}, val={val_fold_df.shape}")
        
        context['ti'].xcom_push(key='split_type', value='stratified')
        context['ti'].xcom_push(key='num_folds', value=num_folds)
        context['ti'].xcom_push(key='fold_paths', value=fold_paths)
        
        return f"Stratified K-Fold split complete: {num_folds} folds created"
        
    elif fold_type == "timeseries":
        # Time Series Split with gap and expanding/sliding window support
        print(f"Performing Time Series Split with {num_folds} folds")
        print(f"  - Gap: {ts_gap} samples")
        print(f"  - Expanding window: {ts_expanding}")
        
        if ts_expanding:
            # Use sklearn TimeSeriesSplit with gap support (expanding window)
            tscv = TimeSeriesSplit(n_splits=num_folds, gap=ts_gap)
            
            fold_paths = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = (y.iloc[train_idx], y.iloc[val_idx]) if y is not None else (None, None)
                
                train_fold_df = pd.concat([X_train_fold, y_train_fold], axis=1) if y_train_fold is not None else X_train_fold
                val_fold_df = pd.concat([X_val_fold, y_val_fold], axis=1) if y_val_fold is not None else X_val_fold
                
                train_fold_path = os.path.join(processed_path, f"train_fold_{fold_idx}.csv")
                val_fold_path = os.path.join(processed_path, f"val_fold_{fold_idx}.csv")
                
                train_fold_df.to_csv(train_fold_path, index=False)
                val_fold_df.to_csv(val_fold_path, index=False)
                
                fold_paths.append({"train": train_fold_path, "val": val_fold_path})
                print(f"Fold {fold_idx} (expanding): train={train_fold_df.shape}, val={val_fold_df.shape}")
        else:
            # Sliding window: custom implementation with fixed train size
            n_samples = len(X)
            # Calculate window size to allow num_folds splits
            total_available = n_samples - ts_gap
            window_size = total_available // (num_folds + 1)  # +1 to ensure room for validation
            
            if window_size < 1:
                raise ValueError(f"Not enough data for {num_folds} folds with gap={ts_gap}. Need more samples.")
            
            print(f"  - Window size: {window_size} samples")
            
            fold_paths = []
            for fold_idx in range(num_folds):
                # Sliding window logic
                train_start = fold_idx * window_size
                train_end = train_start + window_size
                val_start = train_end + ts_gap
                val_end = val_start + window_size
                
                if val_end > n_samples:
                    print(f"Skipping fold {fold_idx}: not enough data for validation set")
                    break
                
                X_train_fold = X.iloc[train_start:train_end]
                X_val_fold = X.iloc[val_start:val_end]
                y_train_fold = y.iloc[train_start:train_end] if y is not None else None
                y_val_fold = y.iloc[val_start:val_end] if y is not None else None
                
                train_fold_df = pd.concat([X_train_fold, y_train_fold], axis=1) if y_train_fold is not None else X_train_fold
                val_fold_df = pd.concat([X_val_fold, y_val_fold], axis=1) if y_val_fold is not None else X_val_fold
                
                train_fold_path = os.path.join(processed_path, f"train_fold_{fold_idx}.csv")
                val_fold_path = os.path.join(processed_path, f"val_fold_{fold_idx}.csv")
                
                train_fold_df.to_csv(train_fold_path, index=False)
                val_fold_df.to_csv(val_fold_path, index=False)
                
                fold_paths.append({"train": train_fold_path, "val": val_fold_path})
                print(f"Fold {fold_idx} (sliding): train[{train_start}:{train_end}], gap[{train_end}:{val_start}], val[{val_start}:{val_end}]")
        
        context['ti'].xcom_push(key='split_type', value='timeseries')
        context['ti'].xcom_push(key='num_folds', value=len(fold_paths))
        context['ti'].xcom_push(key='fold_paths', value=fold_paths)
        context['ti'].xcom_push(key='ts_gap', value=ts_gap)
        context['ti'].xcom_push(key='ts_expanding', value=ts_expanding)
        
        return f"Time Series split complete: {len(fold_paths)} folds created (gap={ts_gap}, expanding={ts_expanding})"
    
    else:
        raise ValueError(f"Unknown FOLD_TYPE: {fold_type}. Supported: null, 'kfold', 'stratified', 'timeseries'")


# DAG definition
with DAG(
    dag_id="01_dag_data",
    default_args=config.DEFAULT_DAG_ARGS,
    description="Data pipeline DAG - loads raw data, performs exploratory data analysis with MLflow logging, and creates train/test splits with support for simple, K-Fold, Stratified K-Fold, and Time Series cross-validation strategies",
    schedule=None,  # Manual trigger for now
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "ml-pipeline"],
) as dag:
    
    # Task 1: Load raw data
    raw_data_load = PythonOperator(
        task_id="raw_data_load",
        python_callable=load_raw_data,
        provide_context=True,
        outlets=[RAW_DATA_ASSET],  # This task produces the raw data asset
    )
    
    # Task 2: Perform EDA and log to MLflow
    eda_task = PythonOperator(
        task_id="eda_task",
        python_callable=perform_eda,
        provide_context=True,
    )
    
    # Task 3: Split data
    data_split = PythonOperator(
        task_id="data_split",
        python_callable=split_data,
        provide_context=True,
        outlets=[TRAIN_TEST_SPLIT_ASSET],  # This task produces the train/test split asset
    )
    
    # Task dependencies
    raw_data_load >> eda_task >> data_split
