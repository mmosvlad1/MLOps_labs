"""Test DAG integrity for CI."""

import pytest


def test_dag_import():
    """Verify DAG files have no import errors."""
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    assert (
        len(dag_bag.import_errors) == 0
    ), f"DAG import errors: {dag_bag.import_errors}"


def test_dag_structure():
    """Verify DAG has expected tasks."""
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    dag = dag_bag.get_dag("ml_training_pipeline")
    assert dag is not None, "DAG 'ml_training_pipeline' not found"

    task_ids = [t.task_id for t in dag.tasks]
    expected_tasks = [
        "check_data",
        "prepare_data",
        "train_model",
        "evaluate_and_decide",
        "register_model",
        "notify_failure",
    ]
    for task_id in expected_tasks:
        assert task_id in task_ids, f"Task '{task_id}' not found in DAG"


def test_dag_no_cycles():
    """Verify DAG has no cycles (is a valid DAG)."""
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    dag = dag_bag.get_dag("ml_training_pipeline")
    # If DagBag loaded it without errors, it's acyclic
    assert dag is not None
