# Adult Income MLOps Pipeline

End-to-end MLOps pipeline for the DDSC611 final project — ESLSCA University, Spring 2026.

**Dataset:** UCI Adult Income (Census Income) — binary classification (income >50K vs <=50K)  
**Stack:** DVC · MLflow · scikit-learn · XGBoost · FastAPI · Evidently · Prometheus · Docker

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
```

---

## Quickstart (3 commands)

Requires Python 3.10. Python 3.12+ is not supported by all dependencies.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy dataset files into data/raw/
#    (place adult.data and adult.test from the UCI dataset here)

# 3. Run the full pipeline
dvc repro && uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000`. Test it:

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

---

## Full Setup (from scratch)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd adult-income-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialise DVC

```bash
dvc init
dvc remote add -d myremote /tmp/dvc-remote   # use any local path
```

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

## Running Tests

```bash
# Unit tests + coverage
pytest tests/unit/ tests/data/ --cov=src --cov-report=term-missing

# Integration tests
pytest tests/integration/

# All tests
pytest tests/ --cov=src --cov-report=term-missing
```

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
│   ├── evidently_reports/       # baseline_report.html, drift_report.html
│   └── prometheus/              # prometheus.yml
├── tests/
│   ├── unit/                    # test_preprocessing.py
│   ├── integration/             # test_api.py
│   └── data/                    # test_data_validation.py
├── docs/
│   ├── model_card.md
│   ├── data_card.md
│   └── experiment_log.csv       # exported from MLflow after training
├── models/                      # best_model.pkl (DVC-tracked)
├── dvc.yaml                     # 3-stage pipeline definition
├── dvc.lock                     # reproducibility lock (committed to Git)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt             # pinned versions
```

---

## Exporting MLflow Experiment Log

After training, export all runs:

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(experiment_names=["adult_income_classification"])
runs.to_csv("docs/experiment_log.csv", index=False)
```

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main` and every PR:

| Stage | Tool | Fails if |
|---|---|---|
| Lint | flake8 | Any style violation |
| Unit Tests | pytest + coverage | Coverage < 70% |
| Data Validation | Pandera | Schema violations |
| Train | DVC + MLflow | Pipeline or training failure |
| Model Validation | evaluate.py | F1 < 0.60 |

Branch protection on `main` requires all stages to pass before merging.

---

## Team

| Name | Student ID |
|---|---|
| Mohamed Ehab | 232400520 |

## Acknowledgements

### External Resources

To be completed before final submission.

### AI Assistance

Claude Code and Codex were used as coding aids during development for different tasks. All generated code was reviewed, understood, and adapted by the team. The team takes full responsibility for all submitted code.

**Module:** DDSC611 – Machine Learning Engineering Practices  
**Instructor:** Mohamed Tharwat, PhD, SM IEEE  
**Academic Year:** 2025–2026 (Spring)
