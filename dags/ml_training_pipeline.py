"""Airflow DAG: ML Training Pipeline for credit card fraud detection.

Pipeline: check_data → prepare_data → train_model → evaluate_and_decide
                                                          ├── register_model
                                                          └── notify_failure
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

log = logging.getLogger(__name__)

DATA_PATH = "/opt/airflow/data/raw/creditcard.csv"
MODELS_DIR = "/opt/airflow/data/models"
AUPRC_THRESHOLD = 0.75

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="ML Training Pipeline: Prepare → Train → Evaluate → Register",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
)


def _check_data(**context):
    """Check if raw data exists; download via kagglehub if missing."""
    from pathlib import Path

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        log.info("Data not found at %s — downloading from Kaggle...", DATA_PATH)
        import shutil

        import kagglehub

        downloaded = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        src = next(downloaded.glob("*.csv"))
        data_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, data_path)
        log.info("Downloaded data to %s", data_path)
    else:
        log.info("Data already exists at %s", DATA_PATH)

    context["ti"].xcom_push(key="data_path", value=str(data_path))
    return str(data_path)


def _evaluate_and_decide(**context):
    """Read metrics.json and branch based on AUPRC quality gate."""
    import json
    from pathlib import Path

    metrics_path = Path(MODELS_DIR) / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    auprc = metrics.get("test_auprc", 0.0)
    log.info("Quality gate — test_auprc=%.4f (threshold=%.2f)", auprc, AUPRC_THRESHOLD)

    context["ti"].xcom_push(key="metrics", value=metrics)

    if auprc >= AUPRC_THRESHOLD:
        log.info("Quality gate PASSED — routing to register_model")
        return "register_model"
    else:
        log.warning("Quality gate FAILED — routing to notify_failure")
        return "notify_failure"


def _register_model(**context):
    """Register model in MLflow Model Registry."""
    import os
    from pathlib import Path

    import joblib
    import mlflow
    import mlflow.sklearn

    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_and_decide", key="metrics")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("creditcard-fraud-production")

    model_path = Path(MODELS_DIR) / "model.pkl"
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="airflow_registration"):
        # Log training parameters
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
            }
        )
        # Log metrics from quality gate evaluation
        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="creditcard-fraud-detector",
        )
        log.info("Model registered as 'creditcard-fraud-detector' in MLflow")


def _notify_failure(**context):
    """Log warning about quality gate failure with actual metrics."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_and_decide", key="metrics")

    auprc = metrics.get("test_auprc", 0.0) if metrics else 0.0
    log.warning(
        "Quality gate FAILED: test_auprc=%.4f did not meet threshold=%.2f",
        auprc,
        AUPRC_THRESHOLD,
    )
    if metrics:
        log.warning("Full metrics: %s", metrics)


check_data = PythonOperator(
    task_id="check_data",
    python_callable=_check_data,
    dag=dag,
)

prepare_data = BashOperator(
    task_id="prepare_data",
    bash_command=(
        "cd /opt/airflow && "
        "python -m src.prepare "
        "--input data/raw/creditcard.csv "
        "--output-dir data/prepared"
    ),
    dag=dag,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command=(
        "cd /opt/airflow && "
        "python -m src.train "
        "--model xgboost "
        "--n-estimators 100 "
        "--max-depth 10 "
        "--learning-rate 0.1 "
        "--model-dir data/models"
    ),
    dag=dag,
)

evaluate_and_decide = BranchPythonOperator(
    task_id="evaluate_and_decide",
    python_callable=_evaluate_and_decide,
    dag=dag,
)

register_model = PythonOperator(
    task_id="register_model",
    python_callable=_register_model,
    dag=dag,
)

notify_failure = PythonOperator(
    task_id="notify_failure",
    python_callable=_notify_failure,
    dag=dag,
)

# Task dependencies
check_data >> prepare_data >> train_model >> evaluate_and_decide
evaluate_and_decide >> [register_model, notify_failure]
