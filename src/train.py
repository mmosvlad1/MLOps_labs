import tempfile
from pathlib import Path

import joblib
import click
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EXPERIMENT_NAME = "creditcard-fraud"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "prepared" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "prepared" / "test.csv"


def get_model(
    name: str,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}. Use xgboost, random_forest, or lightgbm")


def save_confusion_matrix(y_true, y_pred, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance(model, feature_names: list, path: Path):
    imp = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 8))
    order = np.argsort(imp)
    ax.barh([feature_names[i] for i in order], imp[order])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@click.command()
@click.option(
    "--model",
    type=click.Choice(["xgboost", "random_forest", "lightgbm"]),
    default="xgboost",
    help="Модель для навчання",
)
@click.option("--n-estimators", default=100, help="Кількість дерев")
@click.option("--max-depth", default=6, help="Макс. глибина дерева")
@click.option("--learning-rate", default=0.1, help="Learning rate")
@click.option(
    "--train-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Шлях до data/prepared/train.csv",
)
@click.option(
    "--test-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Шлях до data/prepared/test.csv",
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Директорія для збереження моделі (data/models)",
)
@click.option("--random-state", default=42, help="Random state")
def main(
    model: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    train_path: Path | None,
    test_path: Path | None,
    model_dir: Path | None,
    random_state: int,
):
    train_path = train_path or DEFAULT_TRAIN
    test_path = test_path or DEFAULT_TEST

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Підготовлені дані не знайдено. Спочатку запустіть: python -m src.prepare"
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "Class"]
    X_train = train_df[feature_cols]
    y_train = train_df["Class"]
    X_test = test_df[feature_cols]
    y_test = test_df["Class"]

    model_obj = get_model(
        model,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    run_name = f"{model}_n{n_estimators}_d{max_depth}_lr{learning_rate}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", model)
        mlflow.set_tag("n_estimators", str(n_estimators))
        mlflow.set_tag("max_depth", str(max_depth))
        mlflow.set_tag("learning_rate", str(learning_rate))

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "model": model,
        })

        model_obj.fit(X_train, y_train)

        # Metrics on train
        y_train_pred = model_obj.predict(X_train)
        y_train_proba = model_obj.predict_proba(X_train)[:, 1]
        train_metrics = {
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "train_auprc": average_precision_score(y_train, y_train_proba),
        }

        # Metrics on test
        y_test_pred = model_obj.predict(X_test)
        y_test_proba = model_obj.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_auprc": average_precision_score(y_test, y_test_proba),
        }

        mlflow.log_metrics({**train_metrics, **test_metrics})

        mlflow.sklearn.log_model(
            model_obj,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_confusion_matrix(y_test, y_test_pred, tmp / "confusion_matrix.png")
            mlflow.log_artifact(str(tmp / "confusion_matrix.png"))
            save_feature_importance(model_obj, feature_cols, tmp / "feature_importance.png")
            mlflow.log_artifact(str(tmp / "feature_importance.png"))

        if model_dir is not None:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_obj, model_dir / "model.joblib")

        print("Training complete. Logged to MLflow.")
        print(f"  Model: {model}")
        print(f"  Train: {train_metrics}")
        print(f"  Test:  {test_metrics}")


if __name__ == "__main__":
    main()
