"""
05_dag_reload_serve - FastAPI Model Serving Reload DAG

This DAG handles model serving reload after promotion:
- Triggered when a model is promoted (PROMOTED_MODEL_ASSET from DAG 04)
- Calls the FastAPI /admin/reload endpoint to hot-reload champion models
- Verifies the new model version is loaded
- Logs reload status to MLflow

The FastAPI service (model-serve container) serves predictions at:
- GET  /health           - Health check
- GET  /health/ready     - Readiness check (models loaded)
- POST /predict/{name}   - Make predictions
- GET  /admin/models     - List loaded models
- POST /admin/reload     - Reload models from MLflow
"""
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any
import json

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from config_load import Config
from mlflow_log import MLFlowLogger

# Import assets
from assets import PROMOTED_MODEL_ASSET

# Load configuration
config = Config.load()


def reload_fastapi_models(**context):
    """
    Call the FastAPI /admin/reload endpoint to reload champion models.

    The FastAPI service discovers all models with the 'champion' alias
    and loads them for serving.
    """
    import urllib.request
    import urllib.error

    ti = context['ti']
    promotion_config = getattr(config, 'PROMOTION', {})

    # Get FastAPI URL from config or use default
    fastapi_url = promotion_config.get('FASTAPI_URL', 'http://model-serve:8000')
    reload_endpoint = f"{fastapi_url}/admin/reload"

    print(f"\n{'='*60}")
    print("RELOADING FASTAPI MODEL SERVICE")
    print(f"{'='*60}")
    print(f"  Endpoint: {reload_endpoint}")

    reload_result = {
        'success': False,
        'endpoint': reload_endpoint,
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


def verify_model_loaded(**context):
    """
    Verify that the champion model is loaded in FastAPI service.
    Calls /admin/models to check loaded models.
    """
    import urllib.request
    import urllib.error

    ti = context['ti']
    reload_result = ti.xcom_pull(task_ids='reload_fastapi_models', key='reload_result')
    promotion_config = getattr(config, 'PROMOTION', {})

    fastapi_url = promotion_config.get('FASTAPI_URL', 'http://model-serve:8000')
    model_name = promotion_config.get('MODEL_NAME', 'ml_pipeline_model')
    models_endpoint = f"{fastapi_url}/admin/models"

    print(f"\n{'='*60}")
    print("VERIFYING MODEL LOADED")
    print(f"{'='*60}")
    print(f"  Endpoint: {models_endpoint}")
    print(f"  Expected model: {model_name}")

    verify_result = {
        'verified': False,
        'model_name': model_name,
        'model_found': False
    }

    # If reload failed, skip verification
    if not reload_result.get('success', False):
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

            for model in models:
                name = model.get('name')
                version = model.get('version')
                model_type = model.get('model_type')
                print(f"    - {name} v{version} ({model_type})")

                if name == model_name:
                    verify_result['model_found'] = True
                    verify_result['model_version'] = version
                    verify_result['model_type'] = model_type

            if verify_result['model_found']:
                verify_result['verified'] = True
                print(f"\n  SUCCESS: Model '{model_name}' is loaded (v{verify_result['model_version']})")
            else:
                print(f"\n  WARNING: Model '{model_name}' not found in loaded models")

    except Exception as e:
        verify_result['error'] = str(e)
        print(f"  ERROR: {e}")

    print(f"{'='*60}\n")

    ti.xcom_push(key='verify_result', value=verify_result)
    return verify_result


def log_reload_results(**context):
    """
    Log reload results to MLflow for traceability.
    """
    ti = context['ti']

    reload_result = ti.xcom_pull(task_ids='reload_fastapi_models', key='reload_result')
    verify_result = ti.xcom_pull(task_ids='verify_model_loaded', key='verify_result')

    # Initialize MLflow logger
    mlflow_tracking_uri = config.MLFLOW.get("TRACKING_URI")
    mlflow_experiment_name = config.MLFLOW.get("EXPERIMENT_NAME")

    logger = MLFlowLogger(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment_name
    )

    dag_run_id = context.get('dag_run').run_id

    run_name = "D5S1_Model_Reload"
    tags = {
        "task_type": "model_reload",
        "dag_id": context.get('dag').dag_id,
        "task_id": context.get('task').task_id,
        "airflow_dag_run_id": dag_run_id,
        "pipeline_step": "D5S1",
        "reload_success": str(reload_result.get('success', False))
    }

    try:
        logger.start_run(run_name=run_name, tags=tags)

        # Log parameters
        promotion_config = getattr(config, 'PROMOTION', {})
        params = {
            "fastapi_url": promotion_config.get('FASTAPI_URL', 'http://model-serve:8000'),
            "model_name": promotion_config.get('MODEL_NAME', 'ml_pipeline_model')
        }
        logger.log_params(params)

        # Log metrics
        metrics = {
            "reload_success": 1 if reload_result.get('success', False) else 0,
            "model_verified": 1 if verify_result.get('verified', False) else 0
        }

        response = reload_result.get('response', {})
        if response:
            metrics["models_loaded"] = response.get('current_count', 0)
            metrics["models_added"] = len(response.get('added', []))
            metrics["models_updated"] = len(response.get('updated', []))
            metrics["models_removed"] = len(response.get('removed', []))

        logger.log_metrics(metrics)

        # Create summary report
        report_lines = [
            "=" * 60,
            "MODEL SERVING RELOAD REPORT",
            "=" * 60,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Airflow DAG Run: {dag_run_id}",
            "",
            "RELOAD STATUS:",
            f"  Success: {reload_result.get('success', False)}",
            f"  Endpoint: {reload_result.get('endpoint')}",
        ]

        if reload_result.get('error'):
            report_lines.append(f"  Error: {reload_result.get('error')}")

        if response:
            report_lines.extend([
                "",
                "MODELS:",
                f"  Previous count: {response.get('previous_count')}",
                f"  Current count: {response.get('current_count')}",
                f"  Added: {response.get('added', [])}",
                f"  Updated: {response.get('updated', [])}",
                f"  Removed: {response.get('removed', [])}",
            ])

        report_lines.extend([
            "",
            "VERIFICATION:",
            f"  Model: {verify_result.get('model_name')}",
            f"  Found: {verify_result.get('model_found', False)}",
            f"  Version: {verify_result.get('model_version', 'N/A')}",
            f"  Verified: {verify_result.get('verified', False)}",
            "=" * 60
        ])

        logger.log_text("\n".join(report_lines), "reload_report.txt")

        logger.end_run(status="FINISHED")

        print("\n" + "\n".join(report_lines))

        status = "SUCCESS" if reload_result.get('success') and verify_result.get('verified') else "PARTIAL"
        return f"Reload logged: {status}"

    except Exception as e:
        print(f"Error logging reload results: {e}")
        logger.end_run(status="FAILED")
        raise


# DAG definition
with DAG(
    dag_id="05_dag_reload_serve",
    default_args=config.DEFAULT_DAG_ARGS,
    description="FastAPI model serving reload DAG - reloads champion models after promotion",
    schedule=[PROMOTED_MODEL_ASSET],  # Trigger when 04_dag_promote promotes a model
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["model-serving", "ml-pipeline", "fastapi"],
) as dag:

    # Task 1: Reload FastAPI models
    reload_task = PythonOperator(
        task_id="reload_fastapi_models",
        python_callable=reload_fastapi_models,
    )

    # Task 2: Verify model is loaded
    verify_task = PythonOperator(
        task_id="verify_model_loaded",
        python_callable=verify_model_loaded,
    )

    # Task 3: Log reload results
    log_results_task = PythonOperator(
        task_id="log_reload_results",
        python_callable=log_reload_results,
    )

    # Task dependencies
    reload_task >> verify_task >> log_results_task
