-- Initialize MLOps Dashboard database
-- This script creates the database and user for the dashboard application

-- Create dashboard user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'dashboard') THEN
        CREATE USER dashboard WITH PASSWORD 'dashboard_pwd';
    END IF;
END
$$;

-- Create dashboard database
SELECT 'CREATE DATABASE mlops_dashboard'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlops_dashboard')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlops_dashboard TO dashboard;

-- Connect to the dashboard database and create tables
\c mlops_dashboard

-- Create tables (EF Core will handle migrations, but we set up the schema)
CREATE SCHEMA IF NOT EXISTS public;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO dashboard;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dashboard;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dashboard;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO dashboard;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO dashboard;

-- Create experiment_templates table
CREATE TABLE IF NOT EXISTS experiment_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description VARCHAR(1000),
    template_type VARCHAR(50) NOT NULL DEFAULT 'full_pipeline',
    default_config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create generated_experiments table
CREATE TABLE IF NOT EXISTS generated_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID NOT NULL REFERENCES experiment_templates(id) ON DELETE RESTRICT,
    name VARCHAR(255) NOT NULL UNIQUE,
    config_snapshot JSONB,
    generated_path VARCHAR(500),
    status VARCHAR(50) NOT NULL DEFAULT 'generated',
    last_airflow_run_id VARCHAR(255),
    error_message VARCHAR(2000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_generated_experiments_template_id ON generated_experiments(template_id);
CREATE INDEX IF NOT EXISTS idx_generated_experiments_status ON generated_experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiment_templates_template_type ON experiment_templates(template_type);

-- Insert default template (generic baseline)
-- Available preprocessing transformers:
--   imputer:   {"columns": [...], "numeric_strategy": "mean|median", "categorical_strategy": "mode", "fill_value": null}
--   scaler:    {"columns": [...]}
--   onehot:    {"columns": [...]}
--   drop:      {"columns": [...]}
--   dateSplit: {"columns": [...], "dropColumns": true}
--   index:     {"column": "col_name"}
--   sliding_window: {"column": "col_name", "sequence_length": 60}
INSERT INTO experiment_templates (name, description, template_type, default_config)
VALUES (
    'baseline_template',
    'Generic baseline template - customize for your dataset',
    'full_pipeline',
    '{
        "MLFLOW": {
            "TRACKING_URI": "http://mlflow:5000",
            "EXPERIMENT_NAME": "my_experiment"
        },
        "DATA": {
            "RAW_PATH_FILE": "/opt/airflow/data/train.csv",
            "PROCESSED_PATH": "/opt/airflow/data/processed",
            "TRAIN_TEST_SPLIT": 0.2,
            "FOLD_TYPE": null,
            "NUM_FOLDS": 5,
            "STRATIFY_COLUMN": null
        },
        "PREPROCESSING": {
            "FEATURES_PATH": "/opt/airflow/data/features",
            "PIPELINE_SPEC": {
                "steps": []
            }
        },
        "MODEL": {
            "LABEL_COLUMNS": [],
            "EPOCHS": 100,
            "BATCH_SIZE": 32,
            "OPTIMIZER": "adam",
            "EARLY_STOPPING_PATIENCE": 10,
            "VALIDATION_SPLIT": 0.2
        },
        "PROMOTION": {
            "MODEL_NAME": "my_model",
            "CHAMPION_ALIAS": "champion",
            "CHALLENGER_ALIAS": "challenger",
            "COMPARISON_METRIC": "test_accuracy",
            "HIGHER_IS_BETTER": true,
            "MIN_IMPROVEMENT_THRESHOLD": 0.0,
            "AUTO_PROMOTE_FIRST_MODEL": true
        }
    }'::jsonb
)
ON CONFLICT (name) DO NOTHING;

-- MLOps Dashboard database initialized successfully
\echo 'MLOps Dashboard database initialized successfully'
