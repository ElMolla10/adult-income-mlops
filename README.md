# Adult Income MLOps Pipeline

End-to-end MLOps system for the DDSC611 final project — ESLSCA University, Spring 2026.

**Dataset:** UCI Adult Income (Census Income) — binary classification (income >50K vs <=50K)  
**Stack:** DVC · MLflow · scikit-learn · XGBoost · FastAPI · Evidently · Prometheus · Docker · Prefect · Streamlit

---

## Executive Summary

This project turns the UCI Adult Income dataset into a reproducible, observable,
and demo-ready machine learning system. The focus is not only model accuracy:
the repository demonstrates the full model lifecycle, including data/version
control, schema validation, feature engineering, experiment tracking, model
registry promotion, API serving, monitoring, CI/CD gates, and orchestration.

Current production model snapshot:

| Area | Current State |
|---|---|
| Best model | XGBoost |
| Held-out F1 | 0.7097 |
| Accuracy | 0.8541 |
| ROC-AUC | 0.9196 |
| Registry | Current MLflow Production version |
| Tests | 39 passing |
| Coverage | 72.38% whole-`src` coverage |
| Drift action | `RETRAIN_REQUIRED` alert linked to Prefect retraining |

Key engineering choices:

- DVC reproduces the prepare → preprocess → train pipeline.
- MLflow tracks Logistic Regression, Random Forest, and XGBoost runs and stores
  the promoted model version.
- XGBoost trains on the original class distribution and stores a calibrated
  probability threshold for serving.
- FastAPI validates both numeric ranges and categorical domains before
  inference.
- Evidently writes a drift alert that includes the manual Prefect retraining
  command.
- Streamlit presents a polished local demo for predictions, MLflow runs,
  monitoring, API health, model card, and CI/CD status.

---

## Architecture

```
Raw Data (DVC)
    │
    ▼
[Stage 1: prepare]    ── load_data.py ──► data/processed/train.csv + test.csv
    │
    ▼
[Stage 2: preprocess] ── feature_engineering.py ──► pipeline.pkl + splits/
    │
    ▼
[Stage 3: train]      ── train.py ──► 3 MLflow experiments + best_model.pkl
    │                                  └── MLflow Registry (Production)
    ▼
[FastAPI Serving]     ── src/serving/app.py ──► /health /predict /predict/batch
    │
    ▼
[Monitoring]          ── run_monitoring.py ──► 2 Evidently reports + Prometheus
    │
    └── drift_alert.json ──► Prefect deployment trigger
```

---

## Quickstart

Requires Python 3.10 or 3.11. Some pinned dependencies are not compatible with
Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy adult.data and adult.test into data/raw/, then run:
dvc repro && uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

The API is then available at `http://localhost:8000`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 2174, "capital-loss": 0,
    "hours-per-week": 40, "native-country": "United-States"
  }'
```

Invalid categorical values such as `"race": "Alien"` or `"sex": "Dragon"` are
rejected with HTTP 422 before inference.

---

## Full Setup (from scratch)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd adult-income-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure DVC remote

```bash
mkdir -p dvc-remotes/adult-income-mlops
dvc remote modify myremote url dvc-remotes/adult-income-mlops
dvc remote list
```

This repository uses the default DVC remote `myremote` at the repo-local path
`dvc-remotes/adult-income-mlops` as resolved by `dvc remote list`, avoiding
machine-specific absolute paths in the checked-in config.
For a team or cloud remote, point it explicitly at your shared location:
`dvc remote modify myremote url <new-path-or-remote-url>`.

### 3. Add data

```bash
# Copy adult.data and adult.test into data/raw/
dvc add data/raw/adult.data data/raw/adult.test
git add data/raw/.gitignore data/raw/adult.data.dvc data/raw/adult.test.dvc
git commit -m "track raw data with DVC"
```

### 4. Start MLflow tracking server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 5. Run the pipeline

```bash
dvc repro
```

This runs all three stages: prepare → preprocess → train.

### 6. Generate monitoring reports

```bash
python monitoring/run_monitoring.py
# Reports saved to monitoring/evidently_reports/
```

### 7. Start the serving app

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Streamlit Demo App

A self-contained local demo of the trained classifier. It loads the
artifacts directly from disk and does **not** require the MLflow or FastAPI
servers to be running.

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The app provides eight sidebar pages:

- **Pipeline Overview** — metrics, architecture, and live system status.
- **Single Prediction** — 14-field form, example profiles, confidence gauge,
  feature importance chart, and local model prediction.
- **Batch Prediction** — CSV upload, colored results, summary stats, pie chart,
  and downloadable predictions.
- **MLflow Experiments** — compares Logistic Regression, Random Forest, and
  XGBoost from `docs/experiment_log.csv`.
- **Monitoring & Drift Detection** — Evidently report iframe, drift scores,
  `RETRAIN_REQUIRED` alert state, and manual Prefect retraining trigger.
- **Live API Monitor** — FastAPI health, Prometheus metrics, and test request.
- **Model Card** — renders `docs/model_card.md` plus subgroup F1 chart.
- **CI/CD Status** — visual pipeline stages and coverage summary.

If the app reports missing artifacts, run:

```bash
dvc pull
# or, to regenerate locally
dvc repro
```

---

## Docker (Bonus A)

Prerequisite: run the DVC pipeline first so `data/processed/pipeline.pkl` and `models/best_model.pkl` exist locally before building Docker images.

```bash
dvc repro
docker compose up --build

# Test
curl http://localhost:8000/health
curl http://localhost:9090  # Prometheus UI
curl http://localhost:5000  # MLflow UI
```

---

## Prefect Orchestration (Bonus B)

Install orchestration-only dependencies when you want to run the scheduled
training flow locally:

```bash
pip install -r requirements-orchestration.txt
prefect server start
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python orchestration/deploy.py
prefect deployment run "adult_income_training_pipeline/adult-income-weekly"
```

The deployment `adult_income_training_pipeline / adult-income-weekly` runs the
six-stage flow: prepare → validate → preprocess → train → evaluate → register.
The latest demo run completed successfully as `tan-hornet` in the Prefect UI.

When monitoring writes `monitoring/evidently_reports/drift_alert.json` with
`RETRAIN_REQUIRED`, the alert includes the linked Prefect deployment and manual
trigger command:

```bash
prefect deployment run "adult_income_training_pipeline/adult-income-weekly"
```

The Streamlit **Monitoring & Drift Detection** page also shows this command and
provides a button to submit the Prefect retraining run after the Prefect server
and deployment are running.

---

## Running Tests

```bash
# Unit tests
pytest tests/unit/ tests/data/

# Integration tests
pytest tests/integration/

# All tests + whole-src coverage gate
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

## Documentation Artifacts

- [Model Card](docs/model_card.md) — model version, intended use,
  hyperparameters, evaluation metrics, subgroup performance, limitations, and
  ethical guidance.
- [Data Card](docs/data_card.md) — dataset source, schema, splits, class
  distribution, preprocessing decisions, drift simulation, and known biases.
- [Experiment Log](docs/experiment_log.csv) — sanitized MLflow export with
  metrics and hyperparameters for the tracked model families.

---

## Production Hardening Notes

This project is designed as a course-grade, demo-ready MLOps system. Before any
real deployment, the main hardening steps would be:

- Use a separate validation split for model-family selection and reserve the
  final test set for one-time reporting.
- Compare calibrated thresholds against XGBoost `scale_pos_weight`, class
  weighting, or SMOTENC for categorical-aware imbalance handling.
- Add fairness gates for recall/FPR gaps by sex and race before registry
  promotion.
- Route drift alerts to an approval workflow before triggering retraining in
  an unattended environment.
- Replace local DVC/MLflow artifact storage with a shared object store for
  team or cloud deployment.

---

## Repository Structure

```
.
├── .github/workflows/ci.yml     # CI/CD — lint, test, validate, evaluate
├── configs/params.yaml          # All pipeline parameters (no hardcoding)
├── data/
│   ├── raw/                     # DVC-tracked, not in Git
│   ├── processed/               # DVC-tracked
│   └── splits/                  # DVC-tracked
├── src/
│   ├── data/                    # load_data.py, validate_data.py
│   ├── features/                # feature_engineering.py (sklearn Pipeline)
│   ├── training/                # train.py (3 experiments + MLflow)
│   ├── evaluation/              # evaluate.py (CI model validation)
│   └── serving/                 # app.py (FastAPI)
├── monitoring/
│   ├── run_monitoring.py        # Evidently reports + drift threshold logic
│   ├── evidently_reports/       # reports + actionable drift_alert.json
│   └── prometheus/              # prometheus.yml
├── tests/
│   ├── unit/                    # preprocessing + pipeline script tests
│   ├── integration/             # FastAPI contract tests
│   └── data/                    # Pandera validation tests
├── docs/
│   ├── model_card.md
│   ├── data_card.md
│   └── experiment_log.csv       # exported from MLflow after training
├── models/                      # best_model.pkl (DVC-tracked)
├── dvc.yaml                     # 3-stage pipeline definition
├── dvc.lock                     # reproducibility lock (committed to Git)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # pinned core/runtime versions
└── requirements-orchestration.txt # Prefect-only orchestration deps
```

---

## Exporting MLflow Experiment Log

After training, export all runs:

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["adult_income_classification"])
runs.to_csv("docs/experiment_log.csv", index=False)
```

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main` and every PR:

| Stage | Tool | Fails if |
|---|---|---|
| Lint | flake8 | Any style violation |
| Unit Tests | pytest | Unit/data test failure |
| Data Validation | Pandera | Schema violations |
| Integration Tests | pytest | FastAPI contract regression |
| Coverage | pytest-cov | Whole-`src` coverage < 70% |
| Train | DVC + MLflow | Pipeline or training failure |
| Model Validation | evaluate.py | F1 < 0.60 |

The workflow is suitable for branch protection on `main`: lint, unit/data
tests, integration tests, coverage, training, and model validation can all be
required before merging.

---

## Team

| Name | Student ID |
|---|---|
| Mohamed Ehab El Molla | 232400520 |
| Mohamed Atef | 232400048 |
| Yahia Abdelmonaem | 232400649 |

## Acknowledgements

### External Resources

- UCI Adult Income Dataset: https://archive.ics.uci.edu/dataset/2/adult
- MLflow — experiment tracking and model registry: https://mlflow.org
- DVC — data version control and pipeline management: https://dvc.org
- Evidently AI — data drift and model monitoring: https://www.evidentlyai.com
- FastAPI — model serving framework: https://fastapi.tiangolo.com
- scikit-learn — preprocessing and model training: https://scikit-learn.org
- XGBoost — gradient boosting classifier: https://xgboost.readthedocs.io
- Prometheus — metrics collection: https://prometheus.io
- Pandera — data validation: https://pandera.readthedocs.io

### AI Assistance

Claude Code and Codex were used as coding aids during development for different tasks. All generated code was reviewed, understood, and adapted by the team. The team takes full responsibility for all submitted code.

**Module:** DDSC611 – Machine Learning Engineering Practices  
**Instructor:** Mohamed Tharwat, PhD, SM IEEE  
**Academic Year:** 2025–2026 (Spring)
