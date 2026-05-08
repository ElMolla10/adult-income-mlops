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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
from sklearn.model_selection import RandomizedSearchCV
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


def compute_metrics(y_true, y_pred, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


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

        estimator = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=params["preprocessing"]["random_state"])),
                ("model", model),
            ]
        )
        search_grid = {
            f"model__{key}": value
            for key, value in param_grid.items()
        }

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_grid,
            n_iter=params["training"]["hpo_n_iter"],
            cv=params["training"]["hpo_cv"],
            scoring=params["training"]["metric"],
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        mlflow.log_params(
            {key.replace("model__", ""): value for key, value in search.best_params_.items()}
        )

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Log model to MLflow registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=params["serving"]["model_name"],
        )

        run_id = run.info.run_id
        print(
            f"  {model_name:20s} | F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
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
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            {
                "n_estimators": mp["xgboost"]["n_estimators"],
                "max_depth": mp["xgboost"]["max_depth"],
                "learning_rate": mp["xgboost"]["learning_rate"],
                "subsample": mp["xgboost"]["subsample"],
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
