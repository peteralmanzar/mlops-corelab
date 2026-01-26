"""
01_dag_data - Data Pipeline DAG

This DAG handles data-related operations for the ML pipeline.
Currently empty - tasks to be added.
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

# Import assets for data-aware scheduling
from assets import RAW_DATA_ASSET, CLEANED_DATA_ASSET

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
    description="Data pipeline DAG - handles data ingestion and validation",
    schedule=None,  # Manual trigger for now
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "ml-pipeline"],
    outlets=[RAW_DATA_ASSET],  # This DAG produces raw data asset
) as dag:
    
    # Task 1: Load raw data
    raw_data_load = PythonOperator(
        task_id="raw_data_load",
        python_callable=load_raw_data,
        provide_context=True,
        outlets=[RAW_DATA_ASSET],  # This task produces the raw data asset
    )
    
    # Task 2: Split data
    data_split = PythonOperator(
        task_id="data_split",
        python_callable=split_data,
        provide_context=True,
    )
    
    # Task dependencies
    raw_data_load >> data_split
