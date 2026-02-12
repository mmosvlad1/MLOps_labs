import tempfile
from pathlib import Path

import click
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


EXPERIMENT_NAME = "creditcard-fraud"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _time_to_hour(X) -> np.ndarray:
    """Convert Time (seconds) to hour of day (0-23)."""
    arr = np.asarray(X)
    return np.array((arr.ravel() // 3600) % 24, dtype=np.int64).reshape(-1, 1)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(time_col: str = "Time", amount_col: str = "Amount", v_cols: list = None):
    if v_cols is None:
        v_cols = [f"V{i}" for i in range(1, 29)]

    preprocessor = ColumnTransformer(
        [
            ("time", FunctionTransformer(_time_to_hour), [time_col]),
            ("amount", RobustScaler(), [amount_col]),
            ("passthrough", "passthrough", v_cols),
        ]
    )
    return preprocessor


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


def build_pipeline(
    model_name: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    feature_cols: list,
) -> Pipeline:
    time_col = "Time" if "Time" in feature_cols else None
    amount_col = "Amount" if "Amount" in feature_cols else None
    v_cols = [c for c in feature_cols if c.startswith("V")]

    preprocessor = build_preprocessor(time_col=time_col, amount_col=amount_col, v_cols=v_cols)
    model = get_model(
        model_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def get_feature_names_after_preprocess(pipeline: Pipeline, feature_cols: list) -> list:
    ct = pipeline.named_steps["preprocessor"]
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "time":
            names.append("hour_of_day")
        elif name == "amount":
            names.append("Amount_scaled")
        else:
            names.extend(cols)
    return names


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


def save_feature_importance(pipeline: Pipeline, feature_names: list, path: Path):
    model = pipeline.named_steps["model"]
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
    help="Model to train",
)
@click.option("--n-estimators", default=100, help="Number of trees")
@click.option("--max-depth", default=6, help="Max tree depth")
@click.option("--learning-rate", default=0.1, help="Learning rate")
@click.option("--data-path", default=None, help="Path to creditcard.csv (default: data/raw/creditcard.csv)")
@click.option("--random-state", default=42, help="Random state")
def main(model: str, n_estimators: int, max_depth: int, learning_rate: float, data_path: str, random_state: int):
    data_path = data_path or str(PROJECT_ROOT / "data" / "raw" / "creditcard.csv")

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    df = preprocess(df)

    feature_cols = [c for c in df.columns if c != "Class"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pipeline = build_pipeline(
        model_name=model,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        feature_cols=feature_cols,
    )

    run_name = f"{model}_n{n_estimators}_d{max_depth}_lr{learning_rate}"
    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("model", model)
        mlflow.set_tag("n_estimators", str(n_estimators))
        mlflow.set_tag("max_depth", str(max_depth))
        mlflow.set_tag("learning_rate", str(learning_rate))

        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "model": model,
        })

        # Train
        pipeline.fit(X_train, y_train)

        # Predict on train (for overfitting detection)
        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = {
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "train_auprc": average_precision_score(y_train, y_train_proba),
        }

        # Predict on val
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = {
            "val_recall": recall_score(y_val, y_pred, zero_division=0),
            "val_precision": precision_score(y_val, y_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
            "val_auprc": average_precision_score(y_val, y_proba),
        }

        mlflow.log_metrics({**train_metrics, **val_metrics})

        # Log model (Pipeline)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "__main__._time_to_hour",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        # Artifacts
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_confusion_matrix(y_val, y_pred, tmp / "confusion_matrix.png")
            mlflow.log_artifact(str(tmp / "confusion_matrix.png"))

            feature_names = get_feature_names_after_preprocess(pipeline, feature_cols)
            save_feature_importance(pipeline, feature_names, tmp / "feature_importance.png")
            mlflow.log_artifact(str(tmp / "feature_importance.png"))

        print("Training complete. Logged to MLflow.")
        print(f"  Model: {model}")
        print(f"  Train: {train_metrics}")
        print(f"  Val:   {val_metrics}")


if __name__ == "__main__":
    main()
