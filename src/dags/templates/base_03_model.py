"""
Base Model Training DAG Template Factory.

Provides create_model_dag() factory function that creates a model training
DAG with configurable experiment name and config path.

This DAG handles model training operations for the ML pipeline:
- Loads preprocessed data from 02_dag_preprocess
- Auto-detects task type (regression vs binary/multi-class classification)
- Instantiates appropriate model template from config
- Trains models on all splits/folds in parallel
- Logs comprehensive metrics, parameters, and models to MLflow
- Registers the best model to MLflow Model Registry
"""
from datetime import datetime
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import json

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.python import PythonOperator, get_current_context
from airflow.decorators import task

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger
from model_template import (
    GetModelTemplateMLPRegression,
    GetModelTemplateMLPBinaryClassification,
    GetModelTemplateMLPMultiClassification,
    Optimizer
)
from model_builder import ModelBuilder
from hyperparameter_tuner import OptunaHyperparameterTuner


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    transformed_asset = Dataset(f"file://experiments/{experiment_name}/transformed_data.csv")
    trained_asset = Dataset(f"mlflow://experiments/{experiment_name}/model")
    return transformed_asset, trained_asset


def _detect_task_type(y: pd.DataFrame, label_cols: List[str]) -> Tuple[str, int]:
    """
    Auto-detect task type from label data.

    Returns:
        tuple: (task_type, num_classes)
    """
    label_col = label_cols[0]

    if label_col not in y.columns:
        raise ValueError(f"Label column '{label_col}' not found in data. Available: {y.columns.tolist()}")

    y_values = y[label_col].dropna()
    unique_values = y_values.nunique()

    if unique_values < 20 and all(y_values == y_values.astype(int)):
        if unique_values == 2:
            task_type = "binary_classification"
            num_classes = 2
        else:
            task_type = "multi_classification"
            num_classes = unique_values
    else:
        task_type = "regression"
        num_classes = 1

    print(f"Auto-detected task type: {task_type}")
    print(f"  Unique values: {unique_values}")
    print(f"  Value range: [{y_values.min()}, {y_values.max()}]")
    print(f"  Number of classes: {num_classes}")

    return task_type, num_classes


def _metadata_load(config, experiment_name: Optional[str] = None):
    """Create the metadata_load task function with injected config."""
    def metadata_load(**context):
        """Load preprocessed data metadata."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        fold_type = config.DATA.get("FOLD_TYPE")
        split_type = fold_type if fold_type else "simple"

        print(f"{exp_prefix}Split type from config: {split_type}")

        features_path = config.PREPROCESSING.get("FEATURES_PATH", "/opt/airflow/data/features")

        import glob

        folds_info = []

        if split_type == 'simple':
            train_path = os.path.join(features_path, "train_transformed.csv")
            test_path = os.path.join(features_path, "test_transformed.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"Simple split transformed files not found in {features_path}. "
                    f"Expected: train_transformed.csv, test_transformed.csv. "
                    f"Please run 02_dag_preprocess with current config."
                )

            folds_info.append({
                'fold_id': 0,
                'train_path': train_path,
                'test_path': test_path
            })
            print(f"{exp_prefix}Simple split: found transformed files")

        else:
            train_files = sorted(glob.glob(os.path.join(features_path, "train_fold_*_transformed.csv")))
            test_files = sorted(glob.glob(os.path.join(features_path, "test_fold_*_transformed.csv")))

            if len(train_files) == 0:
                raise FileNotFoundError(
                    f"No transformed fold files found in {features_path}. "
                    f"Expected: train_fold_*_transformed.csv files. "
                    f"Config shows FOLD_TYPE={fold_type}. "
                    f"Please run 02_dag_preprocess to generate transformed {fold_type} data."
                )

            num_folds = len(train_files)

            for fold_idx in range(num_folds):
                train_path = os.path.join(features_path, f"train_fold_{fold_idx}_transformed.csv")
                test_path = os.path.join(features_path, f"test_fold_{fold_idx}_transformed.csv")

                if not os.path.exists(train_path) or not os.path.exists(test_path):
                    raise FileNotFoundError(
                        f"Missing transformed files for fold {fold_idx}: {train_path}, {test_path}"
                    )

                folds_info.append({
                    'fold_id': fold_idx,
                    'train_path': train_path,
                    'test_path': test_path
                })

            print(f"{exp_prefix}{split_type} split: found {num_folds} transformed folds")

        # Get the triggering asset events
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

        # Pull invocation_id from metadata_load task
        upstream_dag_id = f"{experiment_name}_02_dag_preprocess" if experiment_name else "02_dag_preprocess"
        invocation_id = None
        if upstream_run_id:
            invocation_id = ti.xcom_pull(dag_id=upstream_dag_id, task_ids='metadata_load', key='invocation_id', run_id=upstream_run_id)

        if not invocation_id:
            invocation_id_list = ti.xcom_pull(dag_id=upstream_dag_id, task_ids='metadata_load', key='invocation_id', include_prior_dates=True)
            if invocation_id_list is not None:
                if isinstance(invocation_id_list, list) and len(invocation_id_list) > 0:
                    invocation_id = invocation_id_list[0]
                elif isinstance(invocation_id_list, str):
                    invocation_id = invocation_id_list

        print(f"{exp_prefix}Pipeline Tag: {invocation_id}")

        if not invocation_id:
            print(f"{exp_prefix}WARNING: No invocation_id found from 02_dag_preprocess!")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            invocation_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        context['ti'].xcom_push(key='invocation_id', value=invocation_id)
        context['ti'].xcom_push(key='split_type', value=split_type)
        context['ti'].xcom_push(key='folds_info', value=folds_info)
        context['ti'].xcom_push(key='num_folds', value=len(folds_info))

        print(f"{exp_prefix}Loaded metadata for {len(folds_info)} fold(s)")

        return folds_info

    return metadata_load


def _model_build(config, experiment_name: Optional[str] = None):
    """Create the model_build task function with injected config."""
    def model_build(**context):
        """Build model configuration by auto-detecting task type."""
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        print(f"{exp_prefix}=== MODEL BUILD STARTING ===")
        ti = context['ti']

        folds_info = ti.xcom_pull(task_ids='metadata_load', key='folds_info')

        if not folds_info:
            raise ValueError("No folds information found from metadata_load task")

        first_fold = folds_info[0]
        train_path = first_fold['train_path']

        print(f"{exp_prefix}Loading data from: {train_path}")
        df = pd.read_csv(train_path)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in df.columns if col not in label_cols]

        X = df[feature_cols]
        y = df[label_cols]

        num_features = X.shape[1]
        num_samples = X.shape[0]

        print(f"{exp_prefix}Data shape: {df.shape}")
        print(f"{exp_prefix}Features: {num_features} columns")
        print(f"{exp_prefix}Samples: {num_samples} rows")

        task_type, num_classes = _detect_task_type(y, label_cols)
        print(f"{exp_prefix}Task type detected: {task_type}, num_classes: {num_classes}")

        optimizer_str = config.MODEL.get("OPTIMIZER", "adam").lower()
        optimizer_map = {
            "adam": Optimizer.ADAM,
            "sgd": Optimizer.SGD,
            "rmsprop": Optimizer.RMS_PROP,
            "adadelta": Optimizer.ADA_DELTA,
            "adagrad": Optimizer.ADA_GRAD,
            "adamax": Optimizer.ADA_MAX,
            "nadam": Optimizer.NADAM,
            "ftrl": Optimizer.FTRL
        }
        optimizer = optimizer_map.get(optimizer_str, Optimizer.ADAM)

        model_config = {
            'task_type': task_type,
            'num_classes': num_classes,
            'num_features': num_features,
            'optimizer': optimizer,
            'optimizer_name': optimizer_str,
            'epochs': config.MODEL.get("EPOCHS", 50),
            'batch_size': config.MODEL.get("BATCH_SIZE", 32),
            'validation_split': config.MODEL.get("VALIDATION_SPLIT", 0.2),
            'early_stopping_patience': config.MODEL.get("EARLY_STOPPING_PATIENCE", 10)
        }

        model_config_serializable = {k: (v.value if hasattr(v, 'value') else v) for k, v in model_config.items()}
        context['ti'].xcom_push(key='model_config', value=model_config_serializable)

        print(f"\n{exp_prefix}Model Configuration:")
        print(json.dumps(model_config_serializable, indent=2))

        return model_config_serializable

    return model_build


def _hyperparameter_tune(config, experiment_name: Optional[str] = None):
    """Create the hyperparameter_tune task function with injected config."""
    def hyperparameter_tune(**context):
        """Run Optuna hyperparameter optimization if enabled."""
        import mlflow
        from sklearn.model_selection import train_test_split

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""
        ti = context['ti']

        # Check if tuning is enabled
        tuning_config = getattr(config, 'HYPERPARAMETER_TUNING', {})
        if not tuning_config.get('ENABLED', False):
            print(f"{exp_prefix}Hyperparameter tuning DISABLED. Using config defaults.")
            context['ti'].xcom_push(key='best_hyperparams', value=None)
            return None

        print(f"\n{'='*60}")
        print(f"{exp_prefix}HYPERPARAMETER TUNING STARTING")
        print(f"{'='*60}")

        # Get model config from previous task
        model_config = ti.xcom_pull(task_ids='model_build', key='model_config')
        folds_info = ti.xcom_pull(task_ids='metadata_load', key='folds_info')

        if not model_config or not folds_info:
            raise ValueError("model_config or folds_info not found from upstream tasks")

        task_type = model_config['task_type']
        num_features = model_config['num_features']
        num_classes = model_config['num_classes']

        # Load training data from first fold
        first_fold = folds_info[0]
        train_path = first_fold['train_path']

        print(f"{exp_prefix}Loading tuning data from: {train_path}")
        train_df = pd.read_csv(train_path)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in train_df.columns if col not in label_cols]

        X = train_df[feature_cols].values
        y = train_df[label_cols].values

        # Split for tuning validation
        val_split = tuning_config.get('VALIDATION_SPLIT_FOR_TUNING', 0.2)
        random_seed = config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=random_seed
        )

        print(f"{exp_prefix}Tuning data: X_train={X_train.shape}, X_val={X_val.shape}")

        # Initialize MLflow for tuning run
        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        invocation_id = ti.xcom_pull(task_ids='metadata_load', key='invocation_id')
        dag_run_id = context.get('dag_run').run_id

        run_name = "D3S2_Hyperparameter_Tuning"
        tags = {
            "task_type": "hyperparameter_tuning",
            "model_type": task_type,
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id if invocation_id else 'unknown',
            "pipeline_step": "D3S2"
        }
        if experiment_name:
            tags["experiment_name"] = experiment_name

        logger.start_run(run_name=run_name, tags=tags)

        try:
            # Log tuning configuration
            logger.log_params({
                'n_trials': tuning_config.get('N_TRIALS', 50),
                'timeout_seconds': tuning_config.get('TIMEOUT_SECONDS', 3600),
                'sampler': tuning_config.get('SAMPLER', 'TPE'),
                'pruner_type': tuning_config.get('PRUNER', {}).get('TYPE', 'MedianPruner'),
                'tuning_epochs': tuning_config.get('TUNING_EPOCHS', 30),
                'task_type': task_type,
                'num_features': num_features,
            })

            # Create tuner and run study
            tuner = OptunaHyperparameterTuner(
                config=config,
                task_type=task_type,
                num_features=num_features,
                num_classes=num_classes,
                mlflow_logger=logger,
                invocation_id=invocation_id,
                experiment_name=experiment_name
            )

            study_name = f"{experiment_name}_hpo_study" if experiment_name else "hpo_study"
            best_params, study = tuner.run_study(
                X_train, y_train, X_val, y_val,
                study_name=study_name
            )

            # Convert to training format
            best_hyperparams = tuner.get_best_params_for_training(best_params)

            # Log best params
            for k, v in best_params.items():
                logger.log_params({f"best_{k}": str(v)})

            # Log study summary metrics
            study_summary = tuner.get_study_summary(study)
            logger.log_metrics({
                'best_val_loss': study_summary['best_val_loss'] or 0.0,
                'n_trials_completed': study_summary['n_trials_completed'],
                'n_trials_pruned': study_summary['n_trials_pruned'],
                'n_trials_failed': study_summary['n_trials_failed'],
            })

            # Log visualizations if enabled
            if tuning_config.get('MLFLOW_TRACKING', {}).get('LOG_VISUALIZATION', True):
                try:
                    import optuna.visualization as vis

                    # Optimization history
                    fig_history = vis.plot_optimization_history(study)
                    mlflow.log_figure(fig_history, "visualizations/optimization_history.html")

                    # Parameter importance (may fail with few trials)
                    if tuning_config.get('MLFLOW_TRACKING', {}).get('LOG_IMPORTANCE', True):
                        try:
                            fig_importance = vis.plot_param_importances(study)
                            mlflow.log_figure(fig_importance, "visualizations/param_importances.html")
                        except Exception as e:
                            print(f"{exp_prefix}Could not generate param importance plot: {e}")
                except Exception as e:
                    print(f"{exp_prefix}Could not generate visualizations: {e}")

            logger.end_run(status="FINISHED")

            # Push best hyperparams to XCom
            context['ti'].xcom_push(key='best_hyperparams', value=best_hyperparams)
            context['ti'].xcom_push(key='tuning_run_id', value=logger.get_run_id())

            print(f"\n{exp_prefix}Hyperparameter tuning completed!")
            print(f"{exp_prefix}Best validation loss: {study.best_value:.6f}")
            print(f"{exp_prefix}Best params:")
            print(json.dumps(best_params, indent=2, default=str))

            return best_hyperparams

        except Exception as e:
            print(f"{exp_prefix}Error during hyperparameter tuning: {e}")
            logger.end_run(status="FAILED")
            raise

    return hyperparameter_tune


def _create_train_fold_model_task(config, experiment_name: Optional[str] = None):
    """Create the train_fold_model task function."""
    @task
    def train_fold_model(fold_info: dict, model_config: dict, best_hyperparams: Optional[dict] = None) -> dict:
        """Train model for a single fold and log to MLflow.

        Args:
            fold_info: Dict with fold_id, train_path, test_path
            model_config: Model configuration from model_build task
            best_hyperparams: Optional hyperparameters from Optuna tuning
        """
        fold_id = fold_info['fold_id']
        train_path = fold_info['train_path']
        test_path = fold_info['test_path']

        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        print(f"\n{'='*60}")
        print(f"{exp_prefix}Training Fold {fold_id}")
        if best_hyperparams:
            print(f"{exp_prefix}Using tuned hyperparameters")
        else:
            print(f"{exp_prefix}Using default hyperparameters")
        print(f"{'='*60}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        feature_cols = [col for col in train_df.columns if col not in label_cols]

        X_train = train_df[feature_cols].values
        y_train = train_df[label_cols].values
        X_test = test_df[feature_cols].values
        y_test = test_df[label_cols].values

        print(f"{exp_prefix}Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"{exp_prefix}Test shape: X={X_test.shape}, y={y_test.shape}")

        task_type = model_config['task_type']
        num_features = model_config['num_features']
        num_classes = model_config['num_classes']

        optimizer_value = model_config['optimizer']
        if isinstance(optimizer_value, str):
            optimizer = Optimizer(optimizer_value)
        else:
            optimizer = optimizer_value

        # Build model with hyperparams if available
        model_builder = ModelBuilder(config)
        model = model_builder.build_model_for_training(
            task_type=task_type,
            num_features=num_features,
            num_classes=num_classes,
            hyperparams=best_hyperparams
        )

        print(f"\n{exp_prefix}Model instantiated: {task_type}")
        model.summary()

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        context = get_current_context()
        ti = context['ti']

        invocation_id = ti.xcom_pull(task_ids='metadata_load', key='invocation_id')
        dag_run_id = context.get('dag_run').run_id

        run_name = f"D3S3_Model_Train_Fold_{fold_id}"
        tags = {
            "task_type": "model_training",
            "model_type": task_type,
            "fold_id": str(fold_id),
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id if invocation_id else 'unknown',
            "pipeline_step": "D3S3",
            "hyperparams_tuned": str(best_hyperparams is not None)
        }

        if experiment_name:
            tags["experiment_name"] = experiment_name

        # Determine training parameters (use tuned params if available)
        if best_hyperparams and 'training' in best_hyperparams:
            train_params = best_hyperparams['training']
            epochs = train_params.get('epochs', model_config['epochs'])
            batch_size = train_params.get('batch_size', model_config['batch_size'])
            early_stopping_patience = train_params.get('early_stopping_patience', model_config['early_stopping_patience'])
            validation_split = train_params.get('validation_split', model_config['validation_split'])
            optimizer_name = train_params.get('optimizer', model_config['optimizer_name'])
            learning_rate = train_params.get('learning_rate', None)
        else:
            epochs = model_config['epochs']
            batch_size = model_config['batch_size']
            early_stopping_patience = model_config['early_stopping_patience']
            validation_split = model_config['validation_split']
            optimizer_name = model_config['optimizer_name']
            learning_rate = None

        try:
            logger.start_run(run_name=run_name, tags=tags)

            params = {
                "fold_id": fold_id,
                "task_type": task_type,
                "num_features": num_features,
                "num_classes": num_classes,
                "optimizer": optimizer_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "validation_split": validation_split,
                "early_stopping_patience": early_stopping_patience,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "hyperparams_tuned": best_hyperparams is not None
            }

            # Add architecture params if tuned
            if best_hyperparams and 'architecture' in best_hyperparams:
                arch = best_hyperparams['architecture']
                params['tuned_num_hidden_layers'] = arch.get('num_hidden_layers')
                params['tuned_hidden_units'] = str(arch.get('hidden_units'))
                params['tuned_dropout_rate'] = arch.get('dropout_rate')
                params['tuned_activation'] = arch.get('activation')

            if learning_rate:
                params['learning_rate'] = learning_rate

            logger.log_params(params)

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )

            print(f"\n{exp_prefix}Training model for {epochs} epochs...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )

            print(f"\n{exp_prefix}Evaluating on test set...")
            test_results = model.evaluate(X_test, y_test, verbose=0)

            final_metrics = {}

            for metric_name in history.history.keys():
                final_value = history.history[metric_name][-1]
                final_metrics[f"final_{metric_name}"] = float(final_value)

            test_metric_names = model.metrics_names
            for i, metric_name in enumerate(test_metric_names):
                final_metrics[f"test_{metric_name}"] = float(test_results[i])

            if task_type in ['binary_classification', 'multi_classification']:
                if len(test_results) > 1:
                    if 'test_accuracy' not in final_metrics:
                        final_metrics['test_accuracy'] = float(test_results[1])

            final_metrics['total_epochs_trained'] = len(history.history['loss'])
            final_metrics['stopped_early'] = int(len(history.history['loss']) < model_config['epochs'])

            logger.log_metrics(final_metrics)

            print(f"\n{exp_prefix}Logging model to MLflow...")
            logger.log_keras_model(
                model=model,
                artifact_path="model"
            )

            history_df = pd.DataFrame(history.history)
            logger.log_dataframe(history_df, "training_history.csv")

            run_id = logger.get_run_id()
            logger.end_run(status="FINISHED")

            results = {
                'fold_id': fold_id,
                'run_id': run_id,
                'task_type': task_type,
                'metrics': final_metrics,
                'model_uri': f"runs:/{run_id}/model"
            }

            print(f"\n{exp_prefix}Fold {fold_id} training completed successfully!")
            print(f"{exp_prefix}Run ID: {run_id}")
            print(f"{exp_prefix}Test {test_metric_names[0]}: {test_results[0]:.4f}")

            return results

        except Exception as e:
            print(f"{exp_prefix}Error during training fold {fold_id}: {e}")
            logger.end_run(status="FAILED")
            raise

    return train_fold_model


def _model_register(config, experiment_name: Optional[str] = None):
    """Create the model_register task function with injected config."""
    def model_register(**context):
        """Aggregate fold results, identify best model, and register to MLflow Model Registry."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        fold_results = ti.xcom_pull(task_ids='train_fold_model')

        if not fold_results:
            raise ValueError("No fold results found from train_fold_model task")

        if isinstance(fold_results, dict):
            fold_results = [fold_results]

        print(f"\n{'='*60}")
        print(f"{exp_prefix}Model Registration - Analyzing {len(fold_results)} fold(s)")
        print(f"{'='*60}")

        task_type = fold_results[0]['task_type']

        if task_type == 'regression':
            best_metric_key = 'test_loss'
            best_fold = min(fold_results, key=lambda x: x['metrics'].get(best_metric_key, float('inf')))
        else:
            test_acc_key = 'test_accuracy'
            final_acc_key = 'final_accuracy'

            sample_metrics = fold_results[0]['metrics']
            if test_acc_key in sample_metrics:
                best_metric_key = test_acc_key
            elif final_acc_key in sample_metrics:
                best_metric_key = final_acc_key
            else:
                best_metric_key = 'test_loss'

            if 'accuracy' in best_metric_key:
                best_fold = max(fold_results, key=lambda x: x['metrics'].get(best_metric_key, 0.0))
            else:
                best_fold = min(fold_results, key=lambda x: x['metrics'].get(best_metric_key, float('inf')))

        print(f"\n{exp_prefix}Best model selection metric: {best_metric_key}")
        print(f"{exp_prefix}Best fold: {best_fold['fold_id']}")

        best_metric_value = best_fold['metrics'].get(best_metric_key)
        if best_metric_value is not None:
            print(f"{exp_prefix}Best {best_metric_key}: {best_metric_value:.4f}")
        else:
            raise KeyError(f"Required metric '{best_metric_key}' not found in fold results.")

        print(f"\n{exp_prefix}All fold results:")
        for result in fold_results:
            fold_id = result['fold_id']
            metric_val = result['metrics'].get(best_metric_key, 'N/A')
            if isinstance(metric_val, (int, float)):
                print(f"  Fold {fold_id}: {best_metric_key}={metric_val:.4f}")
            else:
                print(f"  Fold {fold_id}: {best_metric_key}={metric_val}")

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        # Get preprocessing pipeline run ID
        upstream_dag_id = f"{experiment_name}_02_dag_preprocess" if experiment_name else "02_dag_preprocess"
        pipeline_mlflow_list = ti.xcom_pull(dag_id=upstream_dag_id, task_ids='pipeline_build', key='pipeline_mlflow_run_id', include_prior_dates=True)
        pipeline_mlflow_run_id = pipeline_mlflow_list[0] if pipeline_mlflow_list and isinstance(pipeline_mlflow_list, list) else pipeline_mlflow_list

        invocation_id = ti.xcom_pull(task_ids='metadata_load', key='invocation_id')

        if not pipeline_mlflow_run_id:
            raise ValueError("pipeline_mlflow_run_id not found in XCom. Cannot combine pipeline and model.")

        print(f"{exp_prefix}Combining preprocessing pipeline (mlflow run: {pipeline_mlflow_run_id}) with Keras model (run: {best_fold['run_id']})")

        builder = ModelBuilder(config)
        preprocessing_pipeline = builder.load_preprocessing_pipeline(pipeline_run_id=pipeline_mlflow_run_id, artifact_path='preprocessing_pipeline')
        trained_model = builder.load_trained_model(model_run_id=best_fold['run_id'], artifact_path='model')
        combined_pipeline, combined_path = builder.combine_pipeline_and_model(preprocessing_pipeline, trained_model, save_to_disk=True)

        dag_run_id = context.get('dag_run').run_id
        run_name = "D3S4_Combined_Model_Register"
        tags = {
            "task_type": "combined_pipeline_registration",
            "best_fold_id": str(best_fold['fold_id']),
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "execution_date": str(context.get('execution_date')),
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id,
            "pipeline_step": "D3S4"
        }

        if experiment_name:
            tags["experiment_name"] = experiment_name

        logger.start_run(run_name=run_name, tags=tags)

        try:
            processed_path = config.DATA.get("PROCESSED_PATH") or os.path.join(os.getcwd(), 'data', 'processed')
            sample_df = None
            sample_file = os.path.join(processed_path, 'train.csv')
            if os.path.exists(sample_file):
                sample_df = pd.read_csv(sample_file).head(5)
            input_example = sample_df[[c for c in sample_df.columns if c not in config.MODEL.get('LABEL_COLUMNS', ['target'])]] if sample_df is not None else None
        except Exception:
            input_example = None

        logger.log_sklearn_pipeline(pipeline=combined_pipeline, artifact_path='combined_model', input_example=input_example)
        logger.log_artifact(combined_path, artifact_path='combined_artifacts')

        combined_run_id = logger.get_run_id()
        logger.end_run(status='FINISHED')

        combined_model_uri = f"runs:/{combined_run_id}/combined_model"

        # Use experiment-scoped model name
        promotion_config = getattr(config, 'PROMOTION', {})
        base_model_name = promotion_config.get('MODEL_NAME', 'ml_pipeline_model')
        model_name = f"{experiment_name}_{base_model_name}" if experiment_name else base_model_name

        registration_tags = {
            "task_type": task_type,
            "best_fold_id": str(best_fold['fold_id']),
            "best_metric": best_metric_key,
            "best_metric_value": str(best_metric_value),
            "preprocessing_run_id": pipeline_mlflow_run_id,
            "keras_run_id": best_fold['run_id'],
            "combined_pipeline_path": combined_path,
            "registered_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if experiment_name:
            registration_tags["experiment_name"] = experiment_name

        registered_model = logger.register_model(model_uri=combined_model_uri, model_name=model_name, tags=registration_tags)

        registration_info = {
            'model_name': model_name,
            'model_version': registered_model.version,
            'model_uri': combined_model_uri,
            'best_fold_id': best_fold['fold_id'],
            'best_run_id': best_fold['run_id'],
            'preprocessing_run_id': pipeline_mlflow_run_id,
            'best_metric': best_metric_key,
            'best_metric_value': best_metric_value,
            'task_type': task_type,
            'all_fold_results': fold_results,
            'combined_pipeline_path': combined_path
        }

        context['ti'].xcom_push(key='registration_info', value=registration_info)

        print(f"\n{'='*60}")
        print(f"{exp_prefix}Combined model registered successfully!")
        print(f"  Model: {model_name}")
        print(f"  Version: {registered_model.version}")
        print(f"  Best fold: {best_fold['fold_id']}")
        print(f"  {best_metric_key}: {best_metric_value:.4f}")
        print(f"{'='*60}")

        return f"Registered {model_name} v{registered_model.version} from fold {best_fold['fold_id']}"

    return model_register


def _validate_registered_model(config, experiment_name: Optional[str] = None):
    """Create the validate_registered_model task function with injected config."""
    def validate_registered_model(**context):
        """Validate the registered combined model end-to-end."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        registration_info = ti.xcom_pull(task_ids='model_register', key='registration_info')
        if not registration_info:
            raise ValueError("No registration_info found in XCom from model_register task")

        model_name = registration_info.get('model_name')
        model_version = registration_info.get('model_version')
        invocation_id = ti.xcom_pull(task_ids='metadata_load', key='invocation_id')

        if model_name is None or model_version is None:
            raise ValueError("registration_info missing model_name or model_version")

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
        logger = MLFlowLogger(tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name)

        model_uri = f"models:/{model_name}/{model_version}"
        try:
            combined_pipeline = MLFlowLogger.load_sklearn_pipeline(model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to load registered combined pipeline {model_uri}: {e}")

        processed_path = config.DATA.get("PROCESSED_PATH") or os.path.join(os.getcwd(), 'data', 'processed')
        fold_type = config.DATA.get("FOLD_TYPE")
        split_type = fold_type if fold_type else "simple"

        if split_type == "simple":
            sample_path = os.path.join(processed_path, "train.csv")
        else:
            sample_path = os.path.join(processed_path, "train_fold_0.csv")

        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                f"Sample processed data not found: {sample_path}. "
                f"Split type is {split_type}. Please run 01_dag_data to generate splits."
            )

        print(f"{exp_prefix}Loading validation sample from: {sample_path}")
        sample_df = pd.read_csv(sample_path)
        sample_size = config.MODEL.get("VALIDATION_SAMPLE_SIZE", 100)
        sample_df = sample_df.head(sample_size)

        label_cols = config.MODEL.get("LABEL_COLUMNS", ["target"])
        X_sample = sample_df[[c for c in sample_df.columns if c not in label_cols]]

        dag_run_id = context.get('dag_run').run_id
        run_name = "D3S5_Preprocessed_Model_Validation"
        tags = {
            "task_type": "validation",
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "execution_date": str(context.get('execution_date')),
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id,
            "pipeline_step": "D3S5"
        }

        if experiment_name:
            tags["experiment_name"] = experiment_name

        try:
            start = datetime.now()
            preds = combined_pipeline.transform(X_sample)
            duration = (datetime.now() - start).total_seconds()
        except Exception as e:
            raise RuntimeError(f"Inference with combined pipeline failed: {e}")

        if preds is None:
            raise ValueError("Combined pipeline returned None predictions")
        preds_arr = preds if hasattr(preds, 'shape') else np.asarray(preds)
        if np.isnan(preds_arr).any():
            raise ValueError("Predictions contain NaN values")

        logger.start_run(run_name=run_name, tags=tags)
        logger.log_metrics({
            'validation_sample_size': int(len(X_sample)),
            'inference_time_seconds': float(duration),
            'output_rows': int(preds_arr.shape[0]),
            'output_cols': int(preds_arr.shape[1]) if len(preds_arr.shape) > 1 else 1,
            'inference_success': 1
        })
        logger.log_params({'model_name': model_name, 'model_version': model_version})
        logger.end_run(status='FINISHED')

        return f"Validation successful for {model_name} v{model_version} on {len(X_sample)} samples"

    return validate_registered_model


def create_model_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a model training DAG.

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
    transformed_asset, trained_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_03_dag_model"
    description = f"Model training for {experiment_name} experiment"
    tags = ["model", "training", "ml-pipeline", "experiment", experiment_name]

    # Create DAG
    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[transformed_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    with dag:
        # Task 1: Load preprocessed data metadata from DAG 2
        metadata_load_task = PythonOperator(
            task_id="metadata_load",
            python_callable=_metadata_load(config, experiment_name),
        )

        # Task 2: Build model configuration
        model_build_task = PythonOperator(
            task_id="model_build",
            python_callable=_model_build(config, experiment_name),
        )

        # Task 3: Hyperparameter tuning (optional, runs if HYPERPARAMETER_TUNING.ENABLED=true)
        hyperparameter_tune_task = PythonOperator(
            task_id="hyperparameter_tune",
            python_callable=_hyperparameter_tune(config, experiment_name),
        )

        # Task 4: Train models for all folds in parallel
        train_fold_model = _create_train_fold_model_task(config, experiment_name)
        train_fold_model_tasks = train_fold_model.partial(
            model_config=model_build_task.output,
            best_hyperparams=hyperparameter_tune_task.output
        ).expand(
            fold_info=metadata_load_task.output
        )

        # Task 5: Register best model to MLflow Model Registry
        model_register_task = PythonOperator(
            task_id="model_register",
            python_callable=_model_register(config, experiment_name),
            outlets=[trained_asset],
        )

        # Task 6: Validate registered combined model
        validate_model_task = PythonOperator(
            task_id="validate_model",
            python_callable=_validate_registered_model(config, experiment_name),
        )

        # Task dependencies
        metadata_load_task >> model_build_task >> hyperparameter_tune_task >> train_fold_model_tasks >> model_register_task >> validate_model_task

    return dag
