"""Lab 3: Гіперпараметрична оптимізація з Optuna, MLflow nested runs, Hydra."""
import os
import random

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data(cfg: DictConfig) -> tuple:
    """Завантажує train.csv, test.csv. З train робить train/val split для HPO."""
    train_path = to_absolute_path(cfg.data.train_path)
    test_path = to_absolute_path(cfg.data.test_path)
    target_col = cfg.data.target_col
    val_size = cfg.data.val_size
    seed = cfg.seed

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Колонка '{target_col}' не знайдена.")

    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
    )
    return X_train, X_val, y_train, y_val, X_test, y_test, feature_cols


def build_model(model_type: str, params: dict, seed: int):
    if model_type == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(random_state=seed, eval_metric="logloss", **params)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(random_state=seed, verbose=-1, **params)
    raise ValueError(f"Unknown model.type='{model_type}'.")


def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> dict:
    space = cfg.hpo[model_type]
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "learning_rate": trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=True),
            "subsample": trial.suggest_float("subsample", space.subsample.low, space.subsample.high),
            "colsample_bytree": trial.suggest_float("colsample_bytree", space.colsample_bytree.low, space.colsample_bytree.high),
            "reg_alpha": trial.suggest_float("reg_alpha", space.reg_alpha.low, space.reg_alpha.high, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", space.reg_lambda.low, space.reg_lambda.high, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", space.min_child_weight.low, space.min_child_weight.high),
        }
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "learning_rate": trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=True),
            "num_leaves": trial.suggest_int("num_leaves", space.num_leaves.low, space.num_leaves.high),
            "subsample": trial.suggest_float("subsample", space.subsample.low, space.subsample.high),
            "reg_alpha": trial.suggest_float("reg_alpha", space.reg_alpha.low, space.reg_alpha.high, log=True),
        }
    raise ValueError(f"Unknown model.type='{model_type}'.")


def evaluate(model, X_train, y_train, X_val, y_val, metric: str) -> float:
    model.fit(X_train, y_train)
    if metric == "auprc":
        y_proba = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, y_proba))
    raise ValueError(f"Unknown metric '{metric}'.")


def evaluate_cv(model, X: np.ndarray, y: np.ndarray, metric: str, seed: int, n_splits: int = 5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]
        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_v, y_v, metric))
    return float(np.mean(scores))


def make_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    sampler_name = sampler_name.lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError("sampler має бути: tpe, random")


def objective_factory(model_type: str, cfg: DictConfig, X_train, X_val, y_train, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_type, cfg)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)

            model = build_model(model_type, params=params, seed=cfg.seed)
            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_val], axis=0)
                y = np.concatenate([y_train, y_val], axis=0)
                score = evaluate_cv(model, X, y, cfg.hpo.metric, cfg.seed, cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_val, y_val, cfg.hpo.metric)
            mlflow.log_metric(cfg.hpo.metric, score)
            return score
    return objective


def _run_hpo_for_model(
    model_type: str,
    cfg: DictConfig,
    X_train, X_val, y_train, y_val, X_test, y_test,
    n_trials: int,
) -> None:
    """Запускає HPO для однієї моделі."""
    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed)
    objective = objective_factory(model_type, cfg, X_train, X_val, y_train, y_val)

    run_name = f"hpo_parent_{model_type}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)
        cfg_log = OmegaConf.to_container(cfg, resolve=True)
        cfg_log["model"] = {"type": model_type}
        mlflow.log_dict(cfg_log, "config_resolved.json")

        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(dict(best_trial.params), "best_params.json")

        best_model = build_model(model_type, params=best_trial.params, seed=cfg.seed)
        best_model.fit(X_train, y_train)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        final_auprc = float(average_precision_score(y_test, y_test_proba))
        mlflow.log_metric("final_auprc", final_auprc)

        os.makedirs("models", exist_ok=True)
        model_path = f"models/best_{model_type}.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"[{model_type}] HPO complete. Best {cfg.hpo.metric}: {best_trial.value:.4f}, Final test AUPRC: {final_auprc:.4f}")


def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, y_train, y_val, X_test, y_test, _ = load_data(cfg)

    if getattr(cfg.hpo, "run_all_models", False):
        models = ["xgboost", "random_forest", "lightgbm"]
        n_per_model = getattr(cfg.hpo, "n_trials_per_model", 50)
        print(f"Запуск HPO для {len(models)} моделей, по {n_per_model} trials кожна...")
        for model_type in models:
            print(f"\n--- {model_type.upper()} ---")
            _run_hpo_for_model(
                model_type, cfg,
                X_train, X_val, y_train, y_val, X_test, y_test,
                n_trials=n_per_model,
            )
        print("\nУсі моделі завершено.")
    else:
        _run_hpo_for_model(
            cfg.model.type, cfg,
            X_train, X_val, y_train, y_val, X_test, y_test,
            n_trials=cfg.hpo.n_trials,
        )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
