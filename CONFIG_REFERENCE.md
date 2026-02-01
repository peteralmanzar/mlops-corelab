# Configuration Reference

This document provides a complete reference for all configuration options in MLOps CoreLab.

## Configuration Files

| File | Purpose |
|------|---------|
| `src/config.json` | Your project-specific configuration |
| `src/config.model.json` | Template with all default values |

To customize, copy `config.model.json` to `config.json` and modify as needed. Values in `config.json` override the defaults.

---

## Configuration Sections

- [MLFLOW](#mlflow) - MLflow tracking configuration
- [RANDOM_SEED](#random_seed) - Global random seed
- [ALERT_EMAIL](#alert_email) - Notification emails
- [SLACK_WEBHOOK](#slack_webhook) - Slack notifications
- [DEFAULT_DAG_ARGS](#default_dag_args) - Airflow DAG defaults
- [DATA](#data) - Data paths and splitting configuration
- [PREPROCESSING](#preprocessing) - Preprocessing pipeline specification
- [MODEL](#model) - Model training configuration
- [PROMOTION](#promotion) - Model promotion settings

---

## MLFLOW

MLflow tracking server configuration.

```json
"MLFLOW": {
  "TRACKING_URI": "http://mlflow:5000",
  "EXPERIMENT_NAME": "ml_pipeline"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `TRACKING_URI` | string | `"http://mlflow:5000"` | **Required.** MLflow tracking server URL |
| `EXPERIMENT_NAME` | string | `"ml_pipeline"` | Name of the MLflow experiment for all runs |

---

## RANDOM_SEED

Global random seed for reproducibility.

```json
"RANDOM_SEED": 42
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `RANDOM_SEED` | integer | `42` | Seed for random operations (data splits, model initialization) |

---

## ALERT_EMAIL

Email addresses for pipeline failure notifications.

```json
"ALERT_EMAIL": ["admin@example.com"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ALERT_EMAIL` | array[string] | `["admin@example.com"]` | List of email addresses for alerts |

---

## SLACK_WEBHOOK

Optional Slack webhook for notifications.

```json
"SLACK_WEBHOOK": null
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `SLACK_WEBHOOK` | string \| null | `null` | Slack webhook URL (set `null` to disable) |

---

## DEFAULT_DAG_ARGS

Default arguments applied to all Airflow DAGs.

```json
"DEFAULT_DAG_ARGS": {
  "owner": "data-science-team",
  "depends_on_past": false,
  "email": ["admin@example.com"],
  "email_on_failure": true,
  "email_on_retry": false,
  "retries": 2,
  "retry_delay_seconds": 300
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `owner` | string | `"data-science-team"` | DAG owner identifier |
| `depends_on_past` | boolean | `false` | If `true`, task instances depend on previous run success |
| `email` | array[string] | `["admin@example.com"]` | Email addresses for task notifications |
| `email_on_failure` | boolean | `true` | Send email on task failure |
| `email_on_retry` | boolean | `false` | Send email on task retry |
| `retries` | integer | `2` | Number of retry attempts on failure |
| `retry_delay_seconds` | integer | `300` | **Required.** Delay between retries in seconds (must be >= 0) |

---

## DATA

Data paths, splitting strategy, and validation rules.

```json
"DATA": {
  "RAW_PATH_FILE": "/opt/airflow/data/train.csv",
  "PROCESSED_PATH": "/opt/airflow/data/processed",
  "TRAIN_TEST_SPLIT": 0.2,
  "VALIDATION_RULES": {
    "max_null_percentage": 0.1,
    "min_rows": 100,
    "required_columns": []
  },
  "FOLD_TYPE": null,
  "NUM_FOLDS": 5,
  "KFOLD_SHUFFLE": true,
  "KFOLD_RANDOM_STATE": 42,
  "STRATIFY_COLUMN": "Survived",
  "TIME_COLUMN": null,
  "TIME_SERIES_GAP": 0,
  "TIME_SERIES_EXPANDING": true
}
```

### Basic Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `RAW_PATH_FILE` | string | `"/home/jovyan/data/raw"` | Path to raw input CSV file |
| `PROCESSED_PATH` | string | `"/home/jovyan/data/processed"` | Directory for train/test split files |
| `TRAIN_TEST_SPLIT` | float | `0.2` | Test set proportion (0.0-1.0) |

### Validation Rules

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `VALIDATION_RULES.max_null_percentage` | float | `0.1` | Maximum allowed null percentage per column (0.0-1.0) |
| `VALIDATION_RULES.min_rows` | integer | `100` | Minimum required rows in dataset |
| `VALIDATION_RULES.required_columns` | array[string] | `[]` | List of columns that must exist |

### Cross-Validation Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `FOLD_TYPE` | string \| null | `null` | Split strategy: `null`, `"kfold"`, `"stratified"`, or `"timeseries"` |
| `NUM_FOLDS` | integer | `5` | Number of folds for K-Fold strategies |
| `KFOLD_SHUFFLE` | boolean | `true` | Shuffle data before K-Fold splitting |
| `KFOLD_RANDOM_STATE` | integer | `42` | Random state for K-Fold shuffle |
| `STRATIFY_COLUMN` | string \| null | `null` | Column to stratify by (for simple split or stratified K-Fold) |

### Time Series Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `TIME_COLUMN` | string \| null | `null` | Column to sort by for time series splits |
| `TIME_SERIES_GAP` | integer | `0` | Gap between train and test sets (number of samples) |
| `TIME_SERIES_EXPANDING` | boolean | `true` | If `true`, expanding window; if `false`, sliding window |

### Fold Type Options

| Value | Description |
|-------|-------------|
| `null` | Simple train/test split (uses `TRAIN_TEST_SPLIT` ratio) |
| `"kfold"` | Standard K-Fold cross-validation |
| `"stratified"` | Stratified K-Fold (maintains class distribution) |
| `"timeseries"` | Time Series split (expanding or sliding window) |

---

## PREPROCESSING

Preprocessing pipeline configuration.

```json
"PREPROCESSING": {
  "FEATURES_PATH": "/opt/airflow/data/features",
  "PIPELINE_SPEC": {
    "steps": [
      {"dateSplit": {"columns": [], "dropColumns": true}},
      {"imputer": {"columns": ["Age"], "numeric_strategy": "median"}},
      {"scaler": {"columns": ["Pclass", "SibSp", "Age", "Parch", "Fare"]}},
      {"onehot": {"columns": ["Sex"]}},
      {"drop": {"columns": ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]}},
      {"index": {"column": ""}},
      {"sequencer": {"column": "", "sequence_length": 60}}
    ]
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `FEATURES_PATH` | string | `"/home/jovyan/data/features"` | Directory for transformed feature files |
| `PIPELINE_SPEC` | object | See below | Pipeline step definitions |

### Pipeline Steps

Steps are executed in order. Each step is an object with a single key (the transformer name) and its configuration.

#### dateSplit

Extracts components from datetime columns.

```json
{"dateSplit": {"columns": ["date_column"], "dropColumns": true}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | array[string] | `[]` | Date columns to process |
| `dropColumns` | boolean | `true` | Drop original date columns after extraction |

**Extracted Features:** year, month, day, dayofweek, hour, minute, second

#### imputer

Handles missing values.

```json
{"imputer": {
  "columns": ["Age", "Fare"],
  "numeric_strategy": "median",
  "categorical_strategy": "mode",
  "fill_value": "Unknown"
}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | array[string] | `[]` | Columns to impute |
| `numeric_strategy` | string | `"mean"` | Strategy for numeric columns: `"mean"`, `"median"`, `"constant"` |
| `categorical_strategy` | string | `"mode"` | Strategy for categorical columns: `"mode"`, `"constant"` |
| `fill_value` | string | `"Unknown"` | Fill value when strategy is `"constant"` |

#### scaler

Standardizes numeric features (zero mean, unit variance).

```json
{"scaler": {"columns": ["Age", "Fare", "Pclass"]}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | array[string] | `[]` | Numeric columns to scale |

#### onehot

One-hot encodes categorical features.

```json
{"onehot": {"columns": ["Sex", "Embarked"]}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | array[string] | `[]` | Categorical columns to encode |

**Output:** Creates binary columns for each category (e.g., `Sex_male`, `Sex_female`)

#### drop

Removes columns from the dataset.

```json
{"drop": {"columns": ["PassengerId", "Name", "Ticket"]}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | array[string] | `[]` | Columns to drop |

#### index

Sets a column as the dataframe index.

```json
{"index": {"column": "PassengerId"}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | string | `""` | Column to use as index (empty string to skip) |

#### sequencer

Creates sequences for time series / sequence models.

```json
{"sequencer": {"column": "value", "sequence_length": 60}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | string | `""` | Column to create sequences from (empty to skip) |
| `sequence_length` | integer | `60` | Number of timesteps per sequence |

---

## MODEL

Model training configuration.

```json
"MODEL": {
  "ARTIFACTS_PATH": "/mlflow/artifacts",
  "LABEL_COLUMNS": ["Survived"],
  "EPOCHS": 200,
  "BATCH_SIZE": 50,
  "OPTIMIZER": "adam",
  "EARLY_STOPPING_PATIENCE": 10,
  "VALIDATION_SPLIT": 0.2,
  "PARAMS": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "VALIDATION_THRESHOLDS": {
    "min_accuracy": 0.75,
    "min_precision": 0.7,
    "min_recall": 0.7,
    "min_f1": 0.7,
    "max_inference_time_ms": 1000
  }
}
```

### Training Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ARTIFACTS_PATH` | string | `"/mlflow/artifacts"` | Path for MLflow model artifacts |
| `LABEL_COLUMNS` | array[string] | `["target"]` | Target column name(s) |
| `EPOCHS` | integer | - | Maximum training epochs |
| `BATCH_SIZE` | integer | - | Training batch size |
| `OPTIMIZER` | string | - | Optimizer: `"adam"`, `"sgd"`, `"rmsprop"` |
| `EARLY_STOPPING_PATIENCE` | integer | - | Epochs to wait before early stopping |
| `VALIDATION_SPLIT` | float | - | Validation set proportion during training |

### Model Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `PARAMS.n_estimators` | integer | `100` | Number of estimators (for ensemble models) |
| `PARAMS.max_depth` | integer | `10` | Maximum tree depth (for tree-based models) |
| `PARAMS.random_state` | integer | `42` | Random state for model initialization |

### Validation Thresholds

Thresholds for model validation checks.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `VALIDATION_THRESHOLDS.min_accuracy` | float | `0.75` | Minimum required accuracy |
| `VALIDATION_THRESHOLDS.min_precision` | float | `0.7` | Minimum required precision |
| `VALIDATION_THRESHOLDS.min_recall` | float | `0.7` | Minimum required recall |
| `VALIDATION_THRESHOLDS.min_f1` | float | `0.7` | Minimum required F1 score |
| `VALIDATION_THRESHOLDS.max_inference_time_ms` | integer | `1000` | Maximum allowed inference time in milliseconds |

---

## PROMOTION

Model promotion and serving configuration.

```json
"PROMOTION": {
  "MODEL_NAME": "ml_pipeline_model",
  "CHAMPION_ALIAS": "champion",
  "CHALLENGER_ALIAS": "challenger",
  "COMPARISON_METRIC": "test_accuracy",
  "HIGHER_IS_BETTER": true,
  "MIN_IMPROVEMENT_THRESHOLD": 0.0,
  "AUTO_PROMOTE_FIRST_MODEL": true,
  "FASTAPI_URL": "http://model-serve:8000"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `MODEL_NAME` | string | `"ml_pipeline_model"` | Model name in MLflow registry |
| `CHAMPION_ALIAS` | string | `"champion"` | Alias for production model |
| `CHALLENGER_ALIAS` | string | `"challenger"` | Alias for non-promoted models |
| `COMPARISON_METRIC` | string | `"test_accuracy"` | Metric to compare models |
| `HIGHER_IS_BETTER` | boolean | `true` | Set `false` for loss metrics |
| `MIN_IMPROVEMENT_THRESHOLD` | float | `0.0` | Minimum improvement required to promote |
| `AUTO_PROMOTE_FIRST_MODEL` | boolean | `true` | Auto-promote when no champion exists |
| `FASTAPI_URL` | string | `"http://model-serve:8000"` | URL of FastAPI serving endpoint |

### Comparison Metrics

Common metrics for `COMPARISON_METRIC`:

| Metric | Higher is Better | Use Case |
|--------|------------------|----------|
| `test_accuracy` | Yes | Classification |
| `test_precision` | Yes | Classification |
| `test_recall` | Yes | Classification |
| `test_f1` | Yes | Classification |
| `test_loss` | No | Any |
| `test_mse` | No | Regression |
| `test_mae` | No | Regression |

---

## Example Configurations

### Binary Classification (Titanic)

```json
{
  "config_version": 1,
  "MLFLOW": {
    "TRACKING_URI": "http://mlflow:5000",
    "EXPERIMENT_NAME": "titanic_survival"
  },
  "RANDOM_SEED": 42,
  "DATA": {
    "RAW_PATH_FILE": "/opt/airflow/data/train.csv",
    "PROCESSED_PATH": "/opt/airflow/data/processed",
    "TRAIN_TEST_SPLIT": 0.2,
    "FOLD_TYPE": "stratified",
    "NUM_FOLDS": 5,
    "STRATIFY_COLUMN": "Survived"
  },
  "PREPROCESSING": {
    "FEATURES_PATH": "/opt/airflow/data/features",
    "PIPELINE_SPEC": {
      "steps": [
        {"imputer": {"columns": ["Age"], "numeric_strategy": "median"}},
        {"scaler": {"columns": ["Pclass", "SibSp", "Age", "Parch", "Fare"]}},
        {"onehot": {"columns": ["Sex"]}},
        {"drop": {"columns": ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]}}
      ]
    }
  },
  "MODEL": {
    "LABEL_COLUMNS": ["Survived"],
    "EPOCHS": 200,
    "BATCH_SIZE": 32,
    "EARLY_STOPPING_PATIENCE": 15
  },
  "PROMOTION": {
    "COMPARISON_METRIC": "test_accuracy",
    "HIGHER_IS_BETTER": true
  }
}
```

### Time Series Regression

```json
{
  "config_version": 1,
  "DATA": {
    "RAW_PATH_FILE": "/opt/airflow/data/stock_prices.csv",
    "FOLD_TYPE": "timeseries",
    "NUM_FOLDS": 5,
    "TIME_COLUMN": "date",
    "TIME_SERIES_GAP": 7,
    "TIME_SERIES_EXPANDING": false
  },
  "PREPROCESSING": {
    "PIPELINE_SPEC": {
      "steps": [
        {"dateSplit": {"columns": ["date"], "dropColumns": true}},
        {"scaler": {"columns": ["open", "high", "low", "close", "volume"]}},
        {"sequencer": {"column": "close", "sequence_length": 30}}
      ]
    }
  },
  "MODEL": {
    "LABEL_COLUMNS": ["close"],
    "EPOCHS": 100,
    "BATCH_SIZE": 64
  },
  "PROMOTION": {
    "COMPARISON_METRIC": "test_mse",
    "HIGHER_IS_BETTER": false
  }
}
```

### Simple Train/Test Split

```json
{
  "config_version": 1,
  "DATA": {
    "RAW_PATH_FILE": "/opt/airflow/data/dataset.csv",
    "FOLD_TYPE": null,
    "TRAIN_TEST_SPLIT": 0.25
  }
}
```

---

## Configuration Validation

The configuration loader validates:

1. **Required fields**: `MLFLOW.TRACKING_URI` must be set
2. **Type checking**: `retry_delay_seconds` must be an integer >= 0
3. **File paths**: Checked at runtime by each DAG

Invalid configurations will cause DAG failures with descriptive error messages.
