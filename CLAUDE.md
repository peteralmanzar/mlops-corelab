# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOps CoreLab is a Docker Compose-based ML pipeline platform. It orchestrates end-to-end machine learning workflows through 4 sequential Airflow DAGs, with experiment tracking in MLflow and model serving via FastAPI.

## Running the Stack

```bash
docker-compose up -d --build          # Build and start all services
docker-compose --profile flower up -d  # Include Celery Flower monitoring
docker-compose down -v                 # Stop and clean up
```

No Makefile, pyproject.toml, or test suite exists. There are no linting or formatting tools configured.

## Architecture

### 4-Stage Airflow DAG Pipeline

Each stage triggers the next via Airflow Dataset assets:

1. **DAG 01 - Data** (`base_01_data.py`): Load raw CSV, run EDA, split into train/test (simple, k-fold, stratified, or time-series)
2. **DAG 02 - Preprocess** (`base_02_preprocess.py`): Build sklearn Pipeline from config spec, transform all splits in parallel, write sequences to .npy memmap files
3. **DAG 03 - Model** (`base_03_model.py`): Auto-detect task type, optionally tune with Optuna, train Keras models (MLP/LSTM/1D-CNN) across folds in parallel, register best to MLflow
4. **DAG 04 - Promote** (`base_04_promote.py`): Compare challenger vs champion on configured metric, promote if better

A 5th DAG (`base_05_dag_reload_serve.py`) hot-reloads champion models in the FastAPI serving layer.

### Key Directories

- `src/dags/templates/` — DAG template factories (one per stage)
- `src/dags/configs/{experiment}/config.json` — Per-experiment configuration
- `src/utils/` — Core utilities (config loading, transforms, model building, MLflow logging)
- `infrastructure/` — Docker service configs (airflow, model-serve, jupyterlabs, dashboard, initdb)

### Core Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `config_load.py` | Two-tier config: merges `src/config.model.json` (defaults) with experiment overrides |
| `data_pipeline.py` | Builds sklearn Pipeline from JSON spec; `extract_sequencer_step()` removes sequencer at runtime |
| `data_transform.py` | Custom sklearn transformers: imputer, scaler, onehot, dateSplit, dropNullRows, drop, index, sequencer |
| `model_template.py` | Keras architecture templates (MLP, LSTM, 1D-CNN) with dynamic builders for Optuna |
| `model_builder.py` | Selects model template based on task_type + sequence_length; combines preprocessing + model into single Pipeline |
| `mlflow_log.py` | MLflow wrapper for logging params, metrics, artifacts, and model registry operations |
| `hyperparameter_tuner.py` | Optuna-based AutoML with configurable search space |
| `champion_sense.py` | Champion model selection and comparison logic |

### Data Flow

```
Raw CSV → [DAG 01] → processed/{train,test}.csv (or fold CSVs)
       → [DAG 02] → features/{transformed CSVs or .npy memmaps}
       → [DAG 03] → MLflow model registry (best fold)
       → [DAG 04] → Champion alias promotion
```

File naming conventions:
- Simple split: `train_simple_X.npy`, `train_simple_y.npy`
- K-fold: `train_fold_{idx}_X.npy`, `train_fold_{idx}_y.npy`
- .npy format: X is 3D `(sequences, seq_len, features)`, y is 1D `(sequences,)`

### Creating a New Experiment

1. Create `src/dags/configs/{name}/config.json` (override keys from `src/config.model.json`)
2. Create DAG files in `src/dags/` calling template factories with `experiment_name` and `config_path`
3. Place raw data at the path specified in `DATA.RAW_PATH_FILE`

## Critical Patterns

- **Airflow 3.1.6 imports**: Use `from airflow.sdk import task` — `airflow.sdk.task` is an attribute, NOT a submodule
- **Sequencer extraction**: `PipelineSequencer` cannot live inside sklearn Pipeline (2D→3D mismatch). It is extracted via `extract_sequencer_step()` and run as a post-step that streams to disk with numpy memmap
- **Combined model pipeline**: After training, preprocessing pipeline + Keras model are bundled into a single sklearn Pipeline via `combine_pipeline_and_model()` for serving
- **Invocation ID**: Each pipeline run generates a unique ID (`YYYYMMdd_HHMMSS_8char-UUID`) propagated across all DAGs via XCom for traceability
- **Task type auto-detection**: Target column unique values determine binary classification (2), multi-class (>2, <20 integers), or regression

## Infrastructure Services

| Service | Port | Purpose |
|---------|------|---------|
| Airflow | 8080 | DAG management UI |
| MLflow | 5000 | Experiment tracking & model registry |
| PostgreSQL | 5432 | Backend for Airflow, MLflow, Kestra |
| Redis | 6379 | Celery broker |
| FastAPI | 8000 | Model serving (predict, admin/reload) |
| Jupyter | 8888 | Interactive notebooks |
| Dashboard | 3001 | .NET Blazor experiment management UI |
| Kestra | 8081 | Alternative workflow engine |
