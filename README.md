# MLOps CoreLab

A production-ready Machine Learning Operations (MLOps) pipeline solution that automates the entire ML workflow from data ingestion through model serving.

## Overview

MLOps CoreLab provides a complete, end-to-end ML pipeline with:

- **Automated Workflows**: 5 interconnected Airflow DAGs handling data, preprocessing, training, promotion, and serving
- **Model Versioning**: Full MLflow integration for experiment tracking and model registry
- **Cross-Validation Support**: Simple splits, K-Fold, Stratified K-Fold, and Time Series splits
- **Model Serving**: FastAPI service with hot-reload capabilities
- **End-to-End Traceability**: Unique invocation IDs track each pipeline run across all stages

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | Apache Airflow (CeleryExecutor) |
| ML Framework | TensorFlow/Keras |
| Model Registry | MLflow |
| Model Serving | FastAPI |
| Data Processing | Pandas, scikit-learn |
| Databases | PostgreSQL, Redis |
| Infrastructure | Docker Compose |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended

### Starting the Stack

```bash
# Start all services
docker-compose up -d

# Start with build (first time or after changes)
docker-compose up -d --build

# Include Flower for Celery monitoring
docker-compose --profile flower up -d
```

### Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | - |
| Jupyter | http://localhost:8888 | - |
| Model API | http://localhost:8000 | - |
| Kestra | http://localhost:8081 | - |
| Homepage | http://localhost:3000 | - |

## Pipeline Architecture

The solution consists of 5 DAGs that execute sequentially using Airflow's data-aware scheduling (Assets):

```
01_dag_data → 02_dag_preprocess → 03_dag_model → 04_dag_promote → 05_dag_reload_serve
```

### DAG 01: Data Pipeline

Loads raw data, performs exploratory data analysis, and creates train/test splits.

**Tasks:**
1. `raw_data_load` - Load CSV and capture metadata
2. `eda_task` - Exploratory data analysis with MLflow logging
3. `data_split` - Create train/test splits (supports multiple strategies)

**Supported Split Strategies:**

| Strategy | Config Value | Description |
|----------|--------------|-------------|
| Simple | `null` | Single train/test split (default 80/20) |
| K-Fold | `"kfold"` | K-Fold cross-validation |
| Stratified | `"stratified"` | Maintains class distribution across folds |
| Time Series | `"timeseries"` | Expanding or sliding window splits |

### DAG 02: Preprocessing Pipeline

Builds and applies a sklearn preprocessing pipeline to all data splits.

**Tasks:**
1. `metadata_load` - Detect split type and validate files
2. `pipeline_build` - Build preprocessing pipeline from config
3. `transform_prepare` - Prepare splits for parallel processing
4. `split_transform` - Transform each fold (parallel execution)
5. `validate_data` - Validate zero nulls and numeric features
6. `preprocessed_eda` - EDA on preprocessed data

**Available Transformers:**

| Transformer | Purpose |
|-------------|---------|
| `dateSplit` | Extract date components (year, month, day, etc.) |
| `imputer` | Handle missing values (mean, median, mode, constant) |
| `dropNullRows` | Remove rows with null values in specified columns |
| `scaler` | Standardize numeric features |
| `onehot` | One-hot encode categorical features |
| `drop` | Remove unwanted columns |
| `index` | Set dataframe index |
| `sequencer` | Create sequences for time series models |

### DAG 03: Model Training

Trains models on all folds and registers the best performing model.

**Tasks:**
1. `metadata_load` - Load preprocessed data metadata
2. `model_build` - Auto-detect task type and configure model
3. `train_fold_model` - Train one model per fold (parallel)
4. `model_register` - Register best model to MLflow
5. `validate_model` - End-to-end validation

**Auto-Detected Task Types:**

| Condition | Task Type |
|-----------|-----------|
| < 20 unique integer values, 2 classes | Binary Classification |
| < 20 unique integer values, > 2 classes | Multi-class Classification |
| Otherwise | Regression |

**Model Architecture:**
- Dense neural networks (MLP) with configurable layers
- Early stopping to prevent overfitting
- Combined pipeline: preprocessing + Keras model in single artifact

### DAG 04: Model Promotion

Compares new models against the current champion and promotes if better.

**Tasks:**
1. `load_model_metadata` - Load new model version info
2. `get_champion_model` - Retrieve current champion
3. `compare_models` - Compare using configured metric
4. `promote_model` - Update champion/challenger aliases
5. `log_promotion_results` - Log decision to MLflow

**Promotion Logic:**
- First model auto-promoted (configurable)
- Subsequent models compared against champion
- Supports minimum improvement threshold
- Handles both "higher is better" and "lower is better" metrics

### DAG 05: Model Serving Reload

Hot-reloads champion models in the FastAPI service after promotion.

**Tasks:**
1. `reload_fastapi_models` - Trigger model reload
2. `verify_model_loaded` - Confirm model loading
3. `log_reload_results` - Log reload status

## Model Serving API

The FastAPI service provides REST endpoints for model inference.

### Endpoints

```bash
# Health check
GET /health

# Readiness check (models loaded)
GET /health/ready

# Make predictions
POST /predict/{model_name}
Content-Type: application/json
{"inputs": [[feature1, feature2, ...]]}

# List loaded models
GET /admin/models

# Reload champion models
POST /admin/reload
```

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict/ml_pipeline_model \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[3, 0, 22.0, 1, 0, 7.25]]}'
```

## Triggering the Pipeline

### Via Airflow UI

1. Navigate to http://localhost:8080
2. Enable the DAGs (toggle switches)
3. Trigger `01_dag_data` to start the pipeline
4. Subsequent DAGs trigger automatically via Assets

### Via REST API

```bash
curl -X POST http://localhost:8080/api/v2/dags/01_dag_data/dagRuns \
  -H "Content-Type: application/json" \
  -u airflow:airflow \
  -d '{}'
```

### Via CLI

```bash
docker exec airflow-scheduler airflow dags trigger 01_dag_data
```

## MLflow Experiment Structure

All pipeline runs are logged to MLflow with consistent naming:

```
Experiment: "ml_pipeline"
├── D1S1_Data_EDA
├── D1S2_Data_Split
├── D2S1_Preprocess_Pipeline_Build
├── D2S2_Preprocess_Validation
├── D2S3_Preprocess_EDA
├── D3S1_Model_Train_Fold_0
├── D3S1_Model_Train_Fold_1
├── ... (one per fold)
├── D3S2_Combined_Model_Register
├── D3S3_Preprocessed_Model_Validation
├── D4S1_Model_Promotion
└── D5S1_Model_Reload
```

## Traceability

Each pipeline execution receives a unique **Invocation ID** (format: `YYYYMMDD_HHMMSS_uuid`):

- Generated in DAG 01's data load task
- Propagated via XCom to all downstream DAGs
- Tagged on every MLflow run
- Enables end-to-end tracking of any model back to its source data

## Configuration

Configuration is managed through JSON files in `src/`:

| File | Purpose |
|------|---------|
| `config.json` | Your project configuration |
| `config.model.json` | Template with all defaults |

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for detailed configuration documentation.

## Project Structure

```
mlops-corelab/
├── src/
│   ├── dags/                    # Airflow DAG definitions
│   │   ├── 01_dag_data.py
│   │   ├── 02_dag_preprocess.py
│   │   ├── 03_dag_model.py
│   │   ├── 04_dag_promote.py
│   │   ├── 05_dag_reload_serve.py
│   │   └── assets.py
│   ├── utils/                   # Shared utilities
│   │   ├── config_load.py
│   │   ├── mlflow_log.py
│   │   ├── model_template.py
│   │   ├── model_builder.py
│   │   ├── data_pipeline.py
│   │   └── data_transform.py
│   ├── config.json              # Project configuration
│   └── config.model.json        # Configuration template
├── infrastructure/
│   ├── model-serve/             # FastAPI serving application
│   ├── airflow/                 # Airflow customizations
│   └── initdb/                  # Database initialization
├── data/                        # Data directory (mounted)
└── docker-compose.yaml          # Service definitions
```

## Stopping the Stack

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Troubleshooting

### DAG not triggering automatically

- Verify Assets are properly defined in `assets.py`
- Check that upstream DAG completed successfully
- Review Airflow scheduler logs: `docker-compose logs -f airflow-scheduler`

### Model serving returns 503

- Check if models are loaded: `curl http://localhost:8000/admin/models`
- Trigger reload: `curl -X POST http://localhost:8000/admin/reload`
- Verify MLflow has a champion model registered

### Preprocessing validation fails

- Ensure all required columns are present in raw data
- Check that imputation strategy handles all null values
- Review validation logs in MLflow

## License

This project is provided as-is for educational and development purposes.
