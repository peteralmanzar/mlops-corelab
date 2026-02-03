"""
Base Model Promotion DAG Template Factory.

Provides create_promote_dag() factory function that creates a model promotion
DAG with configurable experiment name and config path.

This DAG handles model promotion operations for the ML pipeline:
- Triggered when a new model is registered (TRAINED_MODEL_ASSET)
- Compares new model against current champion
- Promotes better-performing model to champion alias
- Logs all promotion activities to MLflow

Note: Model reload is handled by 00_dag_mlflow_watcher which monitors
MLflow Model Registry for champion alias changes across ALL experiments.
"""
from datetime import datetime
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.standard.operators.python import PythonOperator

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger


def _get_experiment_assets(experiment_name: str):
    """Get experiment-scoped assets."""
    trained_asset = Dataset(f"mlflow://experiments/{experiment_name}/model")
    promoted_asset = Dataset(f"mlflow://experiments/{experiment_name}/champion")
    return trained_asset, promoted_asset


def _load_model_metadata(config, experiment_name: Optional[str] = None):
    """Create the load_model_metadata task function with injected config."""
    def load_model_metadata(**context):
        """Load registration info for the newly registered model version."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        promotion_config = getattr(config, 'PROMOTION', {})
        base_model_name = promotion_config.get('MODEL_NAME', 'ml_pipeline_model')

        # Avoid double-prefixing if base_model_name already starts with experiment_name
        if experiment_name:
            prefix = f"{experiment_name}_"
            if base_model_name.startswith(prefix):
                model_name = base_model_name
            else:
                model_name = f"{prefix}{base_model_name}"
        else:
            model_name = base_model_name

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
        logger = MLFlowLogger(tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name)

        # Get upstream DAG run from triggering asset events
        triggering_events = context.get('triggering_asset_events')
        upstream_run_id = None

        if triggering_events:
            for asset_uri, events in triggering_events.items():
                for event in events:
                    if hasattr(event, 'source_run_id'):
                        upstream_run_id = event.source_run_id
                        break
                if upstream_run_id:
                    break

        # Strategy 1: Pull registration_info from DAG 03 via XCom
        upstream_dag_id = f"{experiment_name}_03_dag_model" if experiment_name else "03_dag_model"
        registration_info = None
        if upstream_run_id:
            registration_info = ti.xcom_pull(
                dag_id=upstream_dag_id,
                task_ids='model_register',
                key='registration_info',
                run_id=upstream_run_id
            )

        # Strategy 2: Fallback to include_prior_dates
        if not registration_info:
            registration_info_list = ti.xcom_pull(
                dag_id=upstream_dag_id,
                task_ids='model_register',
                key='registration_info',
                include_prior_dates=True
            )
            if registration_info_list:
                if isinstance(registration_info_list, list) and len(registration_info_list) > 0:
                    sorted_info = sorted(
                        registration_info_list,
                        key=lambda x: int(x.get('model_version', 0)),
                        reverse=True
                    )
                    registration_info = sorted_info[0]
                    print(f"{exp_prefix}XCom fallback: selected most recent version {registration_info.get('model_version')}")
                else:
                    registration_info = registration_info_list

        # Strategy 3: Query MLflow Model Registry directly
        if not registration_info:
            print(f"{exp_prefix}XCom pull failed, querying MLflow Model Registry directly...")
            latest_version = logger.get_latest_model_version(model_name)

            if latest_version:
                run_metrics = logger.get_model_version_metrics(latest_version['run_id'])

                task_type = 'classification' if 'test_accuracy' in run_metrics else 'regression'
                if task_type == 'classification':
                    best_metric = 'test_accuracy'
                    best_metric_value = run_metrics.get('test_accuracy', run_metrics.get('final_accuracy', 0))
                else:
                    best_metric = 'test_loss'
                    best_metric_value = run_metrics.get('test_loss', run_metrics.get('final_loss', 0))

                registration_info = {
                    'model_name': model_name,
                    'model_version': latest_version['version'],
                    'model_uri': f"models:/{model_name}/{latest_version['version']}",
                    'best_run_id': latest_version['run_id'],
                    'best_metric': best_metric,
                    'best_metric_value': best_metric_value,
                    'task_type': task_type,
                    'source': 'mlflow_registry'
                }
                print(f"{exp_prefix}Found latest model version from registry: v{latest_version['version']}")
            else:
                raise ValueError(f"No registered model versions found for '{model_name}'. Ensure 03_dag_model has run successfully.")

        # Try to get invocation_id from DAG 03 metadata_load
        invocation_id = None
        if upstream_run_id:
            invocation_id = ti.xcom_pull(
                dag_id=upstream_dag_id,
                task_ids='metadata_load',
                key='invocation_id',
                run_id=upstream_run_id
            )
        if not invocation_id:
            invocation_id = ti.xcom_pull(
                dag_id=upstream_dag_id,
                task_ids='metadata_load',
                key='invocation_id',
                include_prior_dates=True
            )
            if isinstance(invocation_id, list) and invocation_id:
                invocation_id = invocation_id[0]

        # Push to XCom for downstream tasks
        ti.xcom_push(key='registration_info', value=registration_info)
        ti.xcom_push(key='new_model_version', value=registration_info['model_version'])
        ti.xcom_push(key='new_model_name', value=registration_info['model_name'])
        ti.xcom_push(key='invocation_id', value=invocation_id or 'unknown')

        print(f"\n{'='*60}")
        print(f"{exp_prefix}New model registered:")
        print(f"  Name: {registration_info['model_name']}")
        print(f"  Version: {registration_info['model_version']}")
        print(f"  Best metric ({registration_info['best_metric']}): {registration_info['best_metric_value']:.4f}")
        print(f"  Task type: {registration_info['task_type']}")
        print(f"  Invocation ID: {invocation_id or 'unknown'}")
        print(f"{'='*60}\n")

        return registration_info

    return load_model_metadata


def _get_champion_model(config, experiment_name: Optional[str] = None):
    """Create the get_champion_model task function with injected config."""
    def get_champion_model(**context):
        """Retrieve the current champion model using the champion alias."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        promotion_config = getattr(config, 'PROMOTION', {})

        base_model_name = promotion_config.get('MODEL_NAME', 'ml_pipeline_model')

        # Avoid double-prefixing if base_model_name already starts with experiment_name
        if experiment_name:
            prefix = f"{experiment_name}_"
            if base_model_name.startswith(prefix):
                model_name = base_model_name
            else:
                model_name = f"{prefix}{base_model_name}"
        else:
            model_name = base_model_name

        champion_alias = promotion_config.get('CHAMPION_ALIAS', 'champion')

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        champion_info = logger.get_model_version_by_alias(model_name, champion_alias)

        if champion_info:
            champion_metrics = logger.get_model_version_metrics(champion_info['run_id'])
            champion_info['metrics'] = champion_metrics
            print(f"\n{exp_prefix}Current champion found:")
            print(f"  Model: {model_name}")
            print(f"  Version: {champion_info['version']}")
            print(f"  Run ID: {champion_info['run_id']}")
            print(f"  Metrics: {champion_metrics}")
        else:
            print(f"\n{exp_prefix}No champion model found (alias '{champion_alias}' not set)")
            print(f"{exp_prefix}This appears to be the first model deployment.")

        ti.xcom_push(key='champion_info', value=champion_info)
        return champion_info

    return get_champion_model


def _compare_models(config, experiment_name: Optional[str] = None):
    """Create the compare_models task function with injected config."""
    def compare_models(**context):
        """Compare the new model against the champion using the configured metric."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        registration_info = ti.xcom_pull(task_ids='load_model_metadata', key='registration_info')
        champion_info = ti.xcom_pull(task_ids='get_champion_model', key='champion_info')

        promotion_config = getattr(config, 'PROMOTION', {})
        comparison_metric = promotion_config.get('COMPARISON_METRIC', 'test_accuracy')
        higher_is_better = promotion_config.get('HIGHER_IS_BETTER', True)
        min_improvement = promotion_config.get('MIN_IMPROVEMENT_THRESHOLD', 0.0)
        auto_promote_first = promotion_config.get('AUTO_PROMOTE_FIRST_MODEL', True)

        new_metric_value = registration_info.get('best_metric_value')
        new_version = registration_info.get('model_version')

        comparison_result = {
            'new_version': new_version,
            'new_metric': new_metric_value,
            'comparison_metric': comparison_metric,
            'should_promote': False,
            'reason': ''
        }

        if champion_info is None:
            if auto_promote_first:
                comparison_result['should_promote'] = True
                comparison_result['reason'] = 'First model registered - auto-promoting to champion'
                comparison_result['champion_version'] = None
                comparison_result['champion_metric'] = None
            else:
                comparison_result['should_promote'] = False
                comparison_result['reason'] = 'No champion exists but AUTO_PROMOTE_FIRST_MODEL is disabled'
        else:
            champion_metrics = champion_info.get('metrics', {})
            champion_metric_value = champion_metrics.get(comparison_metric)

            comparison_result['champion_version'] = champion_info['version']
            comparison_result['champion_metric'] = champion_metric_value

            if champion_metric_value is None:
                comparison_result['should_promote'] = True
                comparison_result['reason'] = f'Champion missing comparison metric ({comparison_metric})'
            else:
                if higher_is_better:
                    improvement = (new_metric_value - champion_metric_value) / abs(champion_metric_value) if champion_metric_value != 0 else 0
                    is_better = new_metric_value > champion_metric_value + min_improvement
                else:
                    improvement = (champion_metric_value - new_metric_value) / abs(champion_metric_value) if champion_metric_value != 0 else 0
                    is_better = new_metric_value < champion_metric_value - min_improvement

                comparison_result['improvement'] = improvement

                if is_better:
                    comparison_result['should_promote'] = True
                    comparison_result['reason'] = (
                        f'New model ({new_metric_value:.4f}) beats champion ({champion_metric_value:.4f}) '
                        f'with {improvement*100:.2f}% improvement'
                    )
                else:
                    comparison_result['should_promote'] = False
                    comparison_result['reason'] = (
                        f'New model ({new_metric_value:.4f}) does not beat champion ({champion_metric_value:.4f}) '
                        f'by required threshold ({min_improvement*100:.1f}%)'
                    )

        print(f"\n{'='*60}")
        print(f"{exp_prefix}MODEL COMPARISON RESULT")
        print(f"{'='*60}")
        print(f"  New model version: {comparison_result['new_version']}")
        print(f"  New model metric ({comparison_metric}): {comparison_result['new_metric']:.4f}")
        if comparison_result.get('champion_version'):
            print(f"  Champion version: {comparison_result['champion_version']}")
            print(f"  Champion metric: {comparison_result.get('champion_metric', 'N/A')}")
        if 'improvement' in comparison_result:
            print(f"  Improvement: {comparison_result['improvement']*100:.2f}%")
        print(f"\n  DECISION: {'PROMOTE' if comparison_result['should_promote'] else 'DO NOT PROMOTE'}")
        print(f"  Reason: {comparison_result['reason']}")
        print(f"{'='*60}\n")

        ti.xcom_push(key='comparison_result', value=comparison_result)
        return comparison_result

    return compare_models


def _promote_model(config, experiment_name: Optional[str] = None):
    """Create the promote_model task function with injected config."""
    def promote_model(**context):
        """Promote the new model to champion if comparison passed."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        comparison_result = ti.xcom_pull(task_ids='compare_models', key='comparison_result')
        registration_info = ti.xcom_pull(task_ids='load_model_metadata', key='registration_info')

        promotion_config = getattr(config, 'PROMOTION', {})
        base_model_name = promotion_config.get('MODEL_NAME', 'ml_pipeline_model')

        # Avoid double-prefixing if base_model_name already starts with experiment_name
        if experiment_name:
            prefix = f"{experiment_name}_"
            if base_model_name.startswith(prefix):
                model_name = base_model_name
            else:
                model_name = f"{prefix}{base_model_name}"
        else:
            model_name = base_model_name

        champion_alias = promotion_config.get('CHAMPION_ALIAS', 'champion')
        challenger_alias = promotion_config.get('CHALLENGER_ALIAS', 'challenger')

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")
        logger = MLFlowLogger(tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment_name)

        promotion_result = {
            'promoted': False,
            'model_name': model_name,
            'new_version': registration_info['model_version'],
            'champion_alias': champion_alias
        }

        if not comparison_result['should_promote']:
            print(f"{exp_prefix}Skipping promotion: {comparison_result['reason']}")
            logger.set_model_alias(model_name, challenger_alias, registration_info['model_version'])
            promotion_result['challenger_alias_set'] = True
            ti.xcom_push(key='promotion_result', value=promotion_result)
            return promotion_result

        success = logger.set_model_alias(
            model_name=model_name,
            alias=champion_alias,
            version=registration_info['model_version']
        )

        if success:
            promotion_result['promoted'] = True
            promotion_result['previous_champion'] = comparison_result.get('champion_version')
            print(f"\n{'='*60}")
            print(f"{exp_prefix}MODEL PROMOTED!")
            print(f"  {model_name} v{registration_info['model_version']} is now the champion")
            if comparison_result.get('champion_version'):
                print(f"  Previous champion: v{comparison_result['champion_version']}")
            print(f"{'='*60}\n")
        else:
            promotion_result['error'] = 'Failed to set champion alias'
            print(f"{exp_prefix}ERROR: Failed to set champion alias")

        ti.xcom_push(key='promotion_result', value=promotion_result)
        return promotion_result

    return promote_model


def _log_promotion_results(config, experiment_name: Optional[str] = None):
    """Create the log_promotion_results task function with injected config."""
    def log_promotion_results(**context):
        """Log all promotion activities and results to MLflow."""
        ti = context['ti']
        exp_prefix = f"[{experiment_name}] " if experiment_name else ""

        registration_info = ti.xcom_pull(task_ids='load_model_metadata', key='registration_info')
        comparison_result = ti.xcom_pull(task_ids='compare_models', key='comparison_result')
        promotion_result = ti.xcom_pull(task_ids='promote_model', key='promotion_result')
        invocation_id = ti.xcom_pull(task_ids='load_model_metadata', key='invocation_id')

        mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
        mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

        logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name
        )

        dag_run_id = context.get('dag_run').run_id

        run_name = "D4S1_Model_Promotion"
        tags = {
            "task_type": "model_promotion",
            "dag_id": context.get('dag').dag_id,
            "task_id": context.get('task').task_id,
            "airflow_dag_run_id": dag_run_id,
            "invocation_id": invocation_id or 'unknown',
            "pipeline_step": "D4S1",
            "model_promoted": str(promotion_result.get('promoted', False))
        }

        if experiment_name:
            tags["experiment_name"] = experiment_name

        try:
            logger.start_run(run_name=run_name, tags=tags)

            params = {
                "model_name": registration_info.get('model_name'),
                "new_version": str(registration_info.get('model_version')),
                "comparison_metric": comparison_result.get('comparison_metric'),
                "champion_version": str(comparison_result.get('champion_version', 'None')),
                "higher_is_better": str(getattr(config, 'PROMOTION', {}).get('HIGHER_IS_BETTER', True)),
                "min_improvement_threshold": str(getattr(config, 'PROMOTION', {}).get('MIN_IMPROVEMENT_THRESHOLD', 0.0))
            }
            logger.log_params(params)

            metrics = {
                "new_model_metric": float(comparison_result.get('new_metric', 0)),
                "promotion_success": 1 if promotion_result.get('promoted', False) else 0
            }

            if comparison_result.get('champion_metric') is not None:
                metrics["champion_model_metric"] = float(comparison_result['champion_metric'])
            if comparison_result.get('improvement') is not None:
                metrics["improvement_pct"] = float(comparison_result['improvement']) * 100

            logger.log_metrics(metrics)

            report_lines = [
                "=" * 60,
                f"MODEL PROMOTION REPORT {f'({experiment_name})' if experiment_name else ''}",
                "=" * 60,
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Pipeline Invocation ID: {invocation_id or 'unknown'}",
                f"Airflow DAG Run: {dag_run_id}",
                "",
                "NEW MODEL:",
                f"  Name: {registration_info.get('model_name')}",
                f"  Version: {registration_info.get('model_version')}",
                f"  Task Type: {registration_info.get('task_type')}",
                f"  Metric ({comparison_result.get('comparison_metric')}): {comparison_result.get('new_metric'):.4f}",
                "",
                "CHAMPION MODEL:",
                f"  Version: {comparison_result.get('champion_version', 'None (first deployment)')}",
                f"  Metric: {comparison_result.get('champion_metric') if comparison_result.get('champion_metric') is not None else 'N/A'}",
                "",
                "COMPARISON RESULT:",
                f"  Should Promote: {comparison_result.get('should_promote')}",
                f"  Reason: {comparison_result.get('reason')}",
            ]

            if comparison_result.get('improvement') is not None:
                report_lines.append(f"  Improvement: {comparison_result['improvement']*100:.2f}%")

            report_lines.extend([
                "",
                "PROMOTION STATUS:",
                f"  Promoted: {promotion_result.get('promoted', False)}",
                f"  Previous Champion: {promotion_result.get('previous_champion', 'N/A')}",
                "",
                "SERVING:",
                f"  Model reload handled by 00_dag_mlflow_watcher",
                f"  Endpoint: http://model-serve:8000/predict/{registration_info.get('model_name')}",
                "=" * 60
            ])

            logger.log_text("\n".join(report_lines), "promotion_report.txt")

            logger.end_run(status="FINISHED")

            print("\n" + "\n".join(report_lines))

            return f"Promotion logged: {'PROMOTED' if promotion_result.get('promoted') else 'NOT PROMOTED'}"

        except Exception as e:
            print(f"{exp_prefix}Error logging promotion results: {e}")
            logger.end_run(status="FAILED")
            raise

    return log_promotion_results


def create_promote_dag(
    experiment_name: str,
    config_path: Optional[str] = None,
) -> DAG:
    """
    Factory function to create a model promotion DAG.

    Args:
        experiment_name: Experiment name for scoping.
        config_path: Optional path to experiment-specific config.json.

    Returns:
        Configured Airflow DAG object.

    Note: Model reload is handled by 00_dag_mlflow_watcher which monitors
    MLflow Model Registry for champion alias changes across ALL experiments.
    """
    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load()

    # Get assets
    trained_asset, promoted_asset = _get_experiment_assets(experiment_name)

    # Determine DAG ID
    dag_id = f"{experiment_name}_04_dag_promote"
    description = f"Model promotion for {experiment_name} experiment"
    tags = ["promotion", "ml-pipeline", "experiment", experiment_name]

    # Create DAG
    dag = DAG(
        dag_id=dag_id,
        default_args=config.DEFAULT_DAG_ARGS,
        description=description,
        schedule=[trained_asset],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=tags,
    )

    with dag:
        # Task 1: Load model metadata from DAG 03
        load_metadata_task = PythonOperator(
            task_id="load_model_metadata",
            python_callable=_load_model_metadata(config, experiment_name),
        )

        # Task 2: Get current champion model
        get_champion_task = PythonOperator(
            task_id="get_champion_model",
            python_callable=_get_champion_model(config, experiment_name),
        )

        # Task 3: Compare models
        compare_task = PythonOperator(
            task_id="compare_models",
            python_callable=_compare_models(config, experiment_name),
        )

        # Task 4: Promote model if better
        promote_task = PythonOperator(
            task_id="promote_model",
            python_callable=_promote_model(config, experiment_name),
            outlets=[promoted_asset],
        )

        # Task 5: Log promotion results
        log_results_task = PythonOperator(
            task_id="log_promotion_results",
            python_callable=_log_promotion_results(config, experiment_name),
        )

        # Task dependencies
        load_metadata_task >> get_champion_task >> compare_task >> promote_task >> log_results_task

    return dag
