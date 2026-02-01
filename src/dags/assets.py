"""
Airflow Assets (Datasets) definitions for ML pipeline data dependencies.

Assets represent data produced and consumed by DAGs, enabling data-aware scheduling.
"""
from airflow.datasets import Dataset

# Data pipeline assets
RAW_DATA_ASSET = Dataset("file://data/raw/raw_data.csv")
TRAIN_TEST_SPLIT_ASSET = Dataset("file://data/processed/train_test_split.csv")

# Pipeline assets
PREPROCESSING_PIPELINE_ASSET = Dataset("file://model_artifacts/preprocessing_pipeline.joblib")
TRANSFORMED_DATA_ASSET = Dataset("file://data/processed/transformed_data.csv")

# Model training assets
TRAINED_MODEL_ASSET = Dataset("mlflow://models/ml_pipeline_model")

# Model serving assets
PROMOTED_MODEL_ASSET = Dataset("mlflow://models/ml_pipeline_model@champion")
