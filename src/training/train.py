"""
Stage 3: train
Runs three MLflow experiments (LogisticRegression, RandomForest, XGBoost)
with RandomizedSearchCV HPO, logs all metrics/params/artifacts, then
promotes the best model to Production in the MLflow Model Registry.
"""
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_splits():
    X_train = pd.read_csv("data/splits/X_train.csv").values
    X_test = pd.read_csv("data/splits/X_test.csv").values
    y_train = pd.read_csv("data/splits/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/splits/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4),
        "positive_rate": round(float(np.mean(y_pred)), 4),
    }


def choose_threshold(y_true, y_prob: np.ndarray, candidates: list[float]) -> float:
    """Pick the threshold with best F1, preferring precision on ties."""
    best_threshold, best_f1, best_precision = 0.5, -1.0, -1.0

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if (f1, precision, threshold) > (best_f1, best_precision, best_threshold):
            best_threshold = float(threshold)
            best_f1 = f1
            best_precision = precision

    return best_threshold


def calibration_split(X_train, y_train, calibration_size: float, random_state: int):
    """Return fit/calibration splits, falling back for tiny datasets."""
    classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(classes)
    n_rows = len(y_train)
    n_calibration = int(np.ceil(n_rows * calibration_size))
    n_fit = n_rows - n_calibration

    can_stratify = (
        0.0 < calibration_size < 1.0
        and n_classes > 1
        and class_counts.min() >= 2
        and n_calibration >= n_classes
        and n_fit >= n_classes
    )
    if not can_stratify:
        return X_train, X_train, y_train, y_train

    return train_test_split(
        X_train,
        y_train,
        test_size=calibration_size,
        random_state=random_state,
        stratify=y_train,
    )


def run_experiment(
    model_name: str,
    model,
    param_grid: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    params: dict,
    experiment_id: str,
):
    """Run one MLflow experiment with HPO and return (f1, run_id, best_model)."""
    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id) as run:
        mlflow.log_param("model_type", model_name)

        training_params = params.get("training", {})
        preprocessing_params = params.get("preprocessing", {})
        X_fit, X_cal, y_fit, y_cal = calibration_split(
            X_train,
            y_train,
            calibration_size=training_params.get("calibration_size", 0.2),
            random_state=preprocessing_params.get("random_state", 42),
        )
        n_iter = min(
            training_params["hpo_n_iter"],
            len(list(ParameterGrid(param_grid))),
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=training_params["hpo_cv"],
            scoring=training_params["metric"],
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_fit, y_fit)
        best_model = search.best_estimator_

        mlflow.log_params(search.best_params_)
        threshold_candidates = training_params.get("threshold_candidates", [0.5])
        calibration_prob = best_model.predict_proba(X_cal)[:, 1]
        decision_threshold = choose_threshold(
            y_cal, calibration_prob, threshold_candidates
        )
        mlflow.log_param("decision_threshold", decision_threshold)

        # Refit on the full training split after selecting model params and threshold.
        best_model.fit(X_train, y_train)
        best_model.decision_threshold_ = decision_threshold
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob, decision_threshold)
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if np.isfinite(value)}
        )

        # Log model to MLflow registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=params["serving"]["model_name"],
        )

        run_id = run.info.run_id
        print(
            f"  {model_name:20s} | F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | Threshold={decision_threshold:.3f}"
        )
        return metrics["f1"], run_id, best_model


def train() -> None:
    params = load_params()
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["training"]["mlflow_tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = params["training"]["experiment_name"]

    try:
        experiment_id = mlflow.create_experiment(exp_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    X_train, X_test, y_train, y_test = load_splits()
    mp = params["training"]["models"]

    experiments = [
        (
            "LogisticRegression",
            LogisticRegression(random_state=42),
            {
                "C": mp["logistic_regression"]["C"],
                "max_iter": mp["logistic_regression"]["max_iter"],
                "solver": mp["logistic_regression"]["solver"],
            },
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": mp["random_forest"]["n_estimators"],
                "max_depth": mp["random_forest"]["max_depth"],
                "min_samples_split": mp["random_forest"]["min_samples_split"],
            },
        ),
        (
            "XGBoost",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            ),
            {
                "n_estimators": mp["xgboost"]["n_estimators"],
                "max_depth": mp["xgboost"]["max_depth"],
                "learning_rate": mp["xgboost"]["learning_rate"],
                "subsample": mp["xgboost"]["subsample"],
                "colsample_bytree": mp["xgboost"]["colsample_bytree"],
                "min_child_weight": mp["xgboost"]["min_child_weight"],
                "reg_lambda": mp["xgboost"]["reg_lambda"],
                "scale_pos_weight": mp["xgboost"]["scale_pos_weight"],
            },
        ),
    ]

    print("\nRunning experiments:")
    best_f1, best_model_obj, best_run_id = 0.0, None, None

    for model_name, model, param_grid in experiments:
        f1, run_id, best_model = run_experiment(
            model_name, model, param_grid,
            X_train, X_test, y_train, y_test,
            params, experiment_id,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_model_obj = best_model
            best_run_id = run_id

    # Save best model locally (used by CI evaluate step without MLflow server)
    os.makedirs("models", exist_ok=True)
    best_model_path = params["preprocessing"]["best_model_path"]
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model_obj, f)

    # Promote the registered version from the best run through Staging first.
    client = MlflowClient()
    model_name = params["serving"]["model_name"]
    versions = client.search_model_versions(f"name = '{model_name}'")
    best_version = next((v.version for v in versions if v.run_id == best_run_id), None)
    if best_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage="Staging",
            archive_existing_versions=True,
        )
        print(f"\nModel v{best_version} transitioned: None -> Staging")
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(
            f"Model v{best_version} transitioned: Staging -> Production "
            f"(F1={best_f1:.4f})"
        )
    else:
        print(f"\nNo registered model version found for best run {best_run_id}.")

    print("Stage 3 (train) complete.")


if __name__ == "__main__":
    train()
