"""
Adult Income Training Pipeline — Prefect Flow (Bonus B)

End-to-end training orchestration with 6 sequential tasks:
  1. prepare_data    — load & clean raw CSVs
  2. validate_data   — Pandera schema check (halts pipeline on failure)
  3. preprocess      — fit sklearn preprocessing Pipeline, save artifacts
  4. train           — 3 MLflow experiments + threshold calibration + register best model
  5. evaluate        — assert F1 >= threshold (quality gate)
  6. register_model  — verify Production-stage transition in MLflow Registry

Failure semantics:
  Any task that raises an exception causes Prefect to mark all downstream
  tasks as Upstream_Failed (they never run). Each task logs structured
  start/end messages so the Prefect UI shows clear progress.

Run locally (one-shot):
    python orchestration/flows/training_flow.py

Schedule (continuous worker):
    python orchestration/deploy.py
"""
import os
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

# ------------------------------------------------------------------ #
# Make the project root importable regardless of where Prefect runs
# this file from. Prefect's worker may execute from a different cwd,
# so we resolve the repo root once at import time.
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------------ #
# Task 1: prepare_data
# ------------------------------------------------------------------ #
@task(name="prepare_data", retries=0, log_prints=True)
def prepare_data_task() -> bool:
    """Stage 1: load raw adult.data / adult.test, fix labels, save processed CSVs."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    from src.data.load_data import prepare_data

    logger.info("[1/6] prepare_data: starting...")
    prepare_data()
    logger.info("[1/6] prepare_data: complete.")
    return True


# ------------------------------------------------------------------ #
# Task 2: validate_data
# ------------------------------------------------------------------ #
@task(name="validate_data", retries=0, log_prints=True)
def validate_data_task(_upstream: bool) -> bool:
    """Stage 2: Pandera schema validation on processed train data."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    import pandas as pd
    from src.data.validate_data import validate_data, load_params

    params = load_params()
    df = pd.read_csv(params["data"]["processed_train"])

    logger.info(f"[2/6] validate_data: checking {len(df)} rows against schema...")
    if not validate_data(df):
        raise ValueError(
            "Schema validation FAILED — data does not match Pandera schema. "
            "Halting pipeline; downstream tasks will not run."
        )

    logger.info(f"[2/6] validate_data: passed on {len(df)} rows.")
    return True


# ------------------------------------------------------------------ #
# Task 3: preprocess
# ------------------------------------------------------------------ #
@task(name="preprocess", retries=0, log_prints=True)
def preprocess_task(_upstream: bool) -> bool:
    """Stage 3: build & fit sklearn preprocessing Pipeline, save artifacts."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    from src.features.feature_engineering import featurize

    logger.info("[3/6] preprocess: building & fitting preprocessing pipeline...")
    featurize()
    logger.info("[3/6] preprocess: complete; pipeline.pkl + splits saved.")
    return True


# ------------------------------------------------------------------ #
# Task 4: train
# ------------------------------------------------------------------ #
@task(name="train", retries=0, log_prints=True)
def train_task(_upstream: bool) -> bool:
    """Stage 4: 3 MLflow experiments with HPO; promote best to Production."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    from src.training.train import train

    logger.info("[4/6] train: running 3 experiments (LogReg, RandomForest, XGBoost)...")
    train()
    logger.info("[4/6] train: complete; best model promoted to Production.")
    return True


# ------------------------------------------------------------------ #
# Task 5: evaluate
# ------------------------------------------------------------------ #
@task(name="evaluate", retries=0, log_prints=True)
def evaluate_task(_upstream: bool) -> bool:
    """Stage 5: quality gate — assert F1 >= min_f1_threshold."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    from src.evaluation.evaluate import evaluate

    logger.info("[5/6] evaluate: checking model against minimum F1 threshold...")
    try:
        evaluate()
    except SystemExit as exc:
        # evaluate.py uses sys.exit(1) on failure; convert to a real exception
        # so Prefect marks the task failed and halts downstream tasks.
        if exc.code not in (None, 0):
            raise ValueError(
                f"Quality gate FAILED (exit code {exc.code}) — "
                f"model F1 below threshold. Halting pipeline."
            ) from exc

    logger.info("[5/6] evaluate: quality gate PASSED.")
    return True


# ------------------------------------------------------------------ #
# Task 6: register_model
# ------------------------------------------------------------------ #
@task(name="register_model", retries=0, log_prints=True)
def register_model_task(_upstream: bool) -> str:
    """Stage 6: verify Production-stage transition in MLflow Registry."""
    logger = get_run_logger()
    os.chdir(PROJECT_ROOT)

    import yaml
    from mlflow.tracking import MlflowClient

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["training"]["mlflow_tracking_uri"]
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = params["serving"]["model_name"]

    logger.info(f"[6/6] register_model: verifying '{model_name}' in Production...")
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        raise ValueError(
            f"No model version found in Production stage for '{model_name}'. "
            f"Registration step failed."
        )

    version = versions[0].version
    logger.info(
        f"[6/6] register_model: SUCCESS — {model_name} v{version} "
        f"is live in Production stage."
    )
    return str(version)


# ------------------------------------------------------------------ #
# Flow definition
# ------------------------------------------------------------------ #
@flow(
    name="adult_income_training_pipeline",
    description="End-to-end MLOps training pipeline. Halts on first task failure.",
    log_prints=True,
)
def training_pipeline() -> str:
    """Sequential 6-task training pipeline.

    Returns:
        The MLflow Production version number of the registered model.
    """
    logger = get_run_logger()
    logger.info("=" * 70)
    logger.info("Adult Income Training Pipeline — START")
    logger.info("=" * 70)

    prepared = prepare_data_task()
    validated = validate_data_task(prepared)
    preprocessed = preprocess_task(validated)
    trained = train_task(preprocessed)
    evaluated = evaluate_task(trained)
    version = register_model_task(evaluated)

    logger.info("=" * 70)
    logger.info(f"Pipeline complete. Model v{version} is live in Production.")
    logger.info("=" * 70)
    return version


if __name__ == "__main__":
    training_pipeline()
