# Pipeline Orchestration with Prefect (Bonus B)

This directory contains the Prefect 3.x flow that orchestrates the end-to-end
training pipeline. It satisfies the rubric's Bonus B requirement (up to +10
points) for pipeline orchestration with Apache Airflow or Prefect.

## Why Prefect?

Prefect was chosen over Airflow for its lightweight setup (no metadata DB,
scheduler, and webserver containers required) and its Pythonic decorator-based
API. The same orchestration semantics are achieved with dramatically less
infrastructure overhead.

## Pipeline Structure

The flow `adult_income_training_pipeline` chains six sequential tasks. Each task
wraps an existing module from `src/`, runs in order, and halts all downstream
tasks if it raises an exception.

| # | Task | Wraps | Purpose |
|---|------|-------|---------|
| 1 | `prepare_data` | `src/data/load_data.py` | Load raw CSVs, fix labels |
| 2 | `validate_data` | `src/data/validate_data.py` | Pandera schema check |
| 3 | `preprocess` | `src/features/feature_engineering.py` | Fit preprocessing pipeline + save splits |
| 4 | `train` | `src/training/train.py` | 3 experiments, threshold calibration, register best |
| 5 | `evaluate` | `src/evaluation/evaluate.py` | Quality gate (F1 >= 0.60) |
| 6 | `register_model` | (verifies in MLflow Registry) | Confirm Production-stage |

## Quickstart

```bash
# 0. (one-time) install Prefect
pip install -r requirements-orchestration.txt

# 1. Start the Prefect server (UI at http://localhost:4200)
prefect server start

# 2. In a SECOND terminal, register the deployment + start the worker
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python orchestration/deploy.py

# 3. Trigger a run from the UI (Deployments tab → Quick run)
#    or from CLI:
prefect deployment run "adult_income_training_pipeline/adult-income-weekly"
```

The flow can also be run as a one-shot script for development:

```bash
python orchestration/flows/training_flow.py
```

## Drift-Triggered Retraining

The monitoring script writes `monitoring/evidently_reports/drift_alert.json`
when production drift exceeds the configured threshold. That alert is linked to
this Prefect deployment:

```json
{
  "action": "RETRAIN_REQUIRED",
  "retraining": {
    "orchestrator": "prefect",
    "deployment": "adult_income_training_pipeline/adult-income-weekly",
    "manual_trigger_command": "prefect deployment run \"adult_income_training_pipeline/adult-income-weekly\""
  }
}
```

For the demo, retraining is manually triggered from either the Prefect UI, the
CLI command above, or the Streamlit **Monitoring & Drift Detection** page. This
keeps the drift-to-retraining loop explicit without auto-promoting a model
without human review.

## Schedule

Once the deployment is registered, the flow runs automatically every
**Sunday at 02:00 UTC** (cron: `0 2 * * 0`). This is configured in
`orchestration/deploy.py` and visible in the Prefect UI.

## Demonstrating Failure Handling

To prove that task failure halts downstream tasks (rubric requirement):

1. Edit `configs/params.yaml` and set `training.min_f1_threshold: 0.99`.
2. Trigger a new run from the UI.
3. The `evaluate` task will fail because the model's real F1 (~0.71) is below 0.99.
4. The `register_model` task is automatically marked `Upstream_Failed` and never runs.
5. Screenshot the Graph View showing tasks 1–4 green, task 5 red, task 6 grey.
6. Restore `min_f1_threshold: 0.60` for normal operation.

## File Layout

```
orchestration/
├── README.md                    # this file
├── __init__.py
├── deploy.py                    # registers + serves the deployment
└── flows/
    ├── __init__.py
    └── training_flow.py         # 6-task @flow definition
```

## Prerequisites

- Prefect 3.x (in `requirements-orchestration.txt`)
- A running MLflow tracking server (the flow reads/writes to it)
- The `data/raw/adult.data` and `adult.test` files must exist (the flow's
  `prepare_data` task reads them)
