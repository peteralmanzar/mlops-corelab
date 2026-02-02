"""
00_dag_mlflow_watcher - MLflow Champion Watcher DAG

This DAG monitors MLflow Model Registry for champion alias changes and
triggers FastAPI model reload when changes are detected.

Features:
- Decoupled from specific experiment DAGs (DAGs 1-4)
- Monitors ALL registered models for champion changes
- Single source of truth for model serving updates
- Polls on a configurable schedule (default: every 5 minutes)

This replaces the asset-based triggering from DAG 5, providing a more
flexible architecture that works across multiple experiments.
"""
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
from typing import Dict, Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

# Add utils and dags to path
dags_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(dags_dir.parent / "utils"))
sys.path.insert(0, str(dags_dir))
from config_load import Config
from mlflow_log import MLFlowLogger
from champion_sense import MlflowChampionSensor

# Load configuration
config = Config.load()


def reload_fastapi_models(**context) -> Dict[str, Any]:
    """
    Call the FastAPI /admin/reload endpoint to reload champion models.

    The FastAPI service discovers all models with the 'champion' alias
    and loads them for serving.
    """
    import urllib.request
    import urllib.error

    ti = context['ti']
    promotion_config = getattr(config, 'PROMOTION', {})

    # Get change details from sensor
    champion_changes = ti.xcom_pull(
        task_ids='sense_champion_changes',
        key='champion_changes'
    ) or {}

    # Get FastAPI URL from config or use default
    fastapi_url = promotion_config.get('FASTAPI_URL', 'http://model-serve:8000')
    reload_endpoint = f"{fastapi_url}/admin/reload"

    print(f"\n{'='*60}")
    print("RELOADING FASTAPI MODEL SERVICE")
    print(f"{'='*60}")
    print(f"  Endpoint: {reload_endpoint}")
    print(f"  Triggered by changes:")
    if champion_changes.get('new_champions'):
        print(f"    New: {champion_changes['new_champions']}")
    if champion_changes.get('updated_champions'):
        print(f"    Updated: {champion_changes['updated_champions']}")
    if champion_changes.get('removed_champions'):
        print(f"    Removed: {champion_changes['removed_champions']}")

    reload_result = {
        'success': False,
        'endpoint': reload_endpoint,
        'trigger_reason': champion_changes,
        'response': None
    }

    try:
        # Create POST request to reload endpoint
        req = urllib.request.Request(
            reload_endpoint,
            method='POST',
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            reload_result['response'] = response_data
            reload_result['success'] = response_data.get('status') == 'success'

            print(f"\n  Reload Response:")
            print(f"    Status: {response_data.get('status')}")
            print(f"    Previous count: {response_data.get('previous_count')}")
            print(f"    Current count: {response_data.get('current_count')}")

            if response_data.get('added'):
                print(f"    Added: {response_data.get('added')}")
            if response_data.get('removed'):
                print(f"    Removed: {response_data.get('removed')}")
            if response_data.get('updated'):
                print(f"    Updated: {response_data.get('updated')}")

            print(f"\n  Loaded models:")
            for name, version in response_data.get('models', {}).items():
                print(f"    - {name} v{version}")

    except urllib.error.HTTPError as e:
        reload_result['error'] = f"HTTP {e.code}: {e.reason}"
        print(f"  ERROR: HTTP {e.code} - {e.reason}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"  Response: {error_body}")
            reload_result['error_detail'] = error_body
        except Exception:
            pass

    except urllib.error.URLError as e:
        reload_result['error'] = f"Connection error: {e.reason}"
        print(f"  ERROR: Could not connect to FastAPI service")
        print(f"  Reason: {e.reason}")
        print(f"  Is the model-serve container running?")

    except Exception as e:
        reload_result['error'] = str(e)
        print(f"  ERROR: {e}")

    print(f"{'='*60}\n")

    ti.xcom_push(key='reload_result', value=reload_result)
    return reload_result


def verify_models_loaded(**context) -> Dict[str, Any]:
    """
    Verify that champion models are loaded in FastAPI service.
    Calls /admin/models to check loaded models.
    """
    import urllib.request
    import urllib.error

    ti = context['ti']
    reload_result = ti.xcom_pull(task_ids='reload_fastapi_models', key='reload_result')
    current_champions = ti.xcom_pull(
        task_ids='sense_champion_changes',
        key='current_champions'
    ) or {}
    promotion_config = getattr(config, 'PROMOTION', {})

    fastapi_url = promotion_config.get('FASTAPI_URL', 'http://model-serve:8000')
    models_endpoint = f"{fastapi_url}/admin/models"

    print(f"\n{'='*60}")
    print("VERIFYING MODELS LOADED")
    print(f"{'='*60}")
    print(f"  Endpoint: {models_endpoint}")
    print(f"  Expected champions: {list(current_champions.keys())}")

    verify_result = {
        'verified': False,
        'expected_models': list(current_champions.keys()),
        'loaded_models': [],
        'missing_models': []
    }

    # If reload failed, skip verification
    if not reload_result or not reload_result.get('success', False):
        print(f"\n  Skipping verification - reload failed")
        verify_result['skipped'] = True
        verify_result['reason'] = 'Reload failed'
        ti.xcom_push(key='verify_result', value=verify_result)
        return verify_result

    try:
        req = urllib.request.Request(models_endpoint)

        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode('utf-8'))

            models = response_data.get('models', [])
            print(f"\n  Loaded models ({response_data.get('count', 0)}):")

            loaded_names = []
            for model in models:
                name = model.get('name')
                version = model.get('version')
                model_type = model.get('model_type')
                print(f"    - {name} v{version} ({model_type})")
                loaded_names.append(name)

            verify_result['loaded_models'] = loaded_names

            # Check which expected models are missing
            for expected_model in current_champions.keys():
                if expected_model not in loaded_names:
                    verify_result['missing_models'].append(expected_model)

            if not verify_result['missing_models']:
                verify_result['verified'] = True
                print(f"\n  SUCCESS: All champion models are loaded")
            else:
                print(f"\n  WARNING: Missing models: {verify_result['missing_models']}")

    except Exception as e:
        verify_result['error'] = str(e)
        print(f"  ERROR: {e}")

    print(f"{'='*60}\n")

    ti.xcom_push(key='verify_result', value=verify_result)
    return verify_result


def log_watcher_results(**context):
    """
    Log watcher results to MLflow for traceability.
    """
    ti = context['ti']

    champion_changes = ti.xcom_pull(
        task_ids='sense_champion_changes',
        key='champion_changes'
    ) or {}
    current_champions = ti.xcom_pull(
        task_ids='sense_champion_changes',
        key='current_champions'
    ) or {}
    reload_result = ti.xcom_pull(
        task_ids='reload_fastapi_models',
        key='reload_result'
    ) or {}
    verify_result = ti.xcom_pull(
        task_ids='verify_models_loaded',
        key='verify_result'
    ) or {}

    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )

    dag_run_id = context.get('dag_run').run_id

    run_name = "D0S1_Champion_Watcher"
    tags = {
        "task_type": "champion_watcher",
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "airflow_dag_run_id": dag_run_id,
        "pipeline_step": "D0S1",
        "reload_success": str(reload_result.get('success', False))
    }

    try:
        logger.start_run(run_name=run_name, tags=tags)

        # Log parameters
        promotion_config = getattr(config, 'PROMOTION', {})
        params = {
            "fastapi_url": promotion_config.get('FASTAPI_URL', 'http://model-serve:8000'),
            "champions_monitored": len(current_champions),
            "new_champions": len(champion_changes.get('new_champions', [])),
            "updated_champions": len(champion_changes.get('updated_champions', [])),
            "removed_champions": len(champion_changes.get('removed_champions', []))
        }
        logger.log_params(params)

        # Log metrics
        metrics = {
            "reload_success": 1 if reload_result.get('success', False) else 0,
            "models_verified": 1 if verify_result.get('verified', False) else 0,
            "models_loaded": len(verify_result.get('loaded_models', [])),
            "models_missing": len(verify_result.get('missing_models', []))
        }

        response = reload_result.get('response', {})
        if response:
            metrics["models_current_count"] = response.get('current_count', 0)
            metrics["models_added"] = len(response.get('added', []))
            metrics["models_updated"] = len(response.get('updated', []))
            metrics["models_removed"] = len(response.get('removed', []))

        logger.log_metrics(metrics)

        # Create summary report
        report_lines = [
            "=" * 60,
            "MLFLOW CHAMPION WATCHER REPORT",
            "=" * 60,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Airflow DAG Run: {dag_run_id}",
            "",
            "CHAMPION CHANGES DETECTED:",
            f"  New: {champion_changes.get('new_champions', [])}",
            f"  Updated: {champion_changes.get('updated_champions', [])}",
            f"  Removed: {champion_changes.get('removed_champions', [])}",
            "",
            "CURRENT CHAMPIONS:",
        ]

        for name, info in current_champions.items():
            report_lines.append(f"  - {name} v{info.get('version')}")

        report_lines.extend([
            "",
            "RELOAD STATUS:",
            f"  Success: {reload_result.get('success', False)}",
            f"  Endpoint: {reload_result.get('endpoint')}",
        ])

        if reload_result.get('error'):
            report_lines.append(f"  Error: {reload_result.get('error')}")

        report_lines.extend([
            "",
            "VERIFICATION:",
            f"  Verified: {verify_result.get('verified', False)}",
            f"  Loaded: {verify_result.get('loaded_models', [])}",
            f"  Missing: {verify_result.get('missing_models', [])}",
            "=" * 60
        ])

        logger.log_text("\n".join(report_lines), "watcher_report.txt")

        logger.end_run(status="FINISHED")

        print("\n" + "\n".join(report_lines))

        status = "SUCCESS" if reload_result.get('success') and verify_result.get('verified') else "PARTIAL"
        return f"Watcher logged: {status}"

    except Exception as e:
        print(f"Error logging watcher results: {e}")
        logger.end_run(status="FAILED")
        raise


# DAG definition
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id="00_dag_mlflow_watcher",
    default_args=default_args,
    description="MLflow Champion Watcher - monitors Model Registry and reloads FastAPI on changes",
    schedule=timedelta(minutes=5),  # Poll every 5 minutes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # Only one instance at a time
    tags=["mlflow", "model-serving", "watcher", "infrastructure"],
) as dag:

    # Task 1: Sense for champion changes
    sense_changes = MlflowChampionSensor(
        task_id="sense_champion_changes",
        tracking_uri=config.MLFLOW.get("TRACKING_URI", "http://mlflow:5000"),
        champion_alias="champion",
        state_variable_key="mlflow_champion_state",
        poke_interval=60,  # Check every 60 seconds within the sensor
        timeout=240,  # Timeout after 4 minutes (before next scheduled run)
        mode="reschedule",  # Free up worker slot while waiting
        soft_fail=True,  # Mark as skipped (not failed) when no changes detected
    )

    # Task 2: Reload FastAPI models
    reload_models = PythonOperator(
        task_id="reload_fastapi_models",
        python_callable=reload_fastapi_models,
    )

    # Task 3: Verify models are loaded
    verify_loaded = PythonOperator(
        task_id="verify_models_loaded",
        python_callable=verify_models_loaded,
    )

    # Task 4: Log results to MLflow
    log_results = PythonOperator(
        task_id="log_watcher_results",
        python_callable=log_watcher_results,
    )

    # Task dependencies
    sense_changes >> reload_models >> verify_loaded >> log_results
