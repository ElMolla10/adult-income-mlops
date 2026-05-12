"""
Professional Streamlit demo for the Adult Income MLOps pipeline.

The app loads the trained model and preprocessing pipeline from disk for local
predictions, then layers in optional live views for FastAPI, Prometheus,
MLflow metadata, Evidently drift reports, and CI/CD status.
"""
from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("adult_income_demo")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import get_decision_threshold

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PIPELINE_PATH = PROJECT_ROOT / "data" / "processed" / "pipeline.pkl"
PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "docs" / "experiment_log.csv"
MODEL_CARD_PATH = PROJECT_ROOT / "docs" / "model_card.md"
BASELINE_REPORT_PATH = PROJECT_ROOT / "monitoring" / "evidently_reports" / "baseline_report.html"
DRIFT_REPORT_PATH = PROJECT_ROOT / "monitoring" / "evidently_reports" / "drift_report.html"
DRIFT_ALERT_PATH = PROJECT_ROOT / "monitoring" / "evidently_reports" / "drift_alert.json"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

API_BASE_URL = os.environ.get("ADULT_INCOME_API_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

MODEL_METRICS = {
    "F1": 0.7097,
    "Accuracy": 0.8541,
    "ROC-AUC": 0.9196,
    "Precision": 0.6697,
    "Recall": 0.7548,
}

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
FEATURE_ORDER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

DISPLAY_LABELS = {
    "age": "Age",
    "workclass": "Employment type",
    "fnlwgt": "Census sample weight",
    "education": "Education level",
    "education-num": "Years of education",
    "marital-status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Household relationship",
    "race": "Race",
    "sex": "Sex",
    "capital-gain": "Capital gain",
    "capital-loss": "Capital loss",
    "hours-per-week": "Hours worked per week",
    "native-country": "Native country",
}
REVERSE_DISPLAY_LABELS = {label: column for column, label in DISPLAY_LABELS.items()}

NUMERIC_RANGES = {
    "age": (17, 90),
    "fnlwgt": (1, 2_000_000),
    "education-num": (1, 16),
    "capital-gain": (0, 100_000),
    "capital-loss": (0, 10_000),
    "hours-per-week": (1, 99),
}

CATEGORY_OPTIONS: dict[str, list[str]] = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
}

CATEGORY_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "workclass": {
        "Self-emp-not-inc": "Self-employed, not incorporated",
        "Self-emp-inc": "Self-employed, incorporated",
        "Federal-gov": "Federal government",
        "Local-gov": "Local government",
        "State-gov": "State government",
        "Without-pay": "Unpaid work",
        "Never-worked": "Never worked",
    },
    "education": {
        "Some-college": "Some college",
        "HS-grad": "High school graduate",
        "Prof-school": "Professional school",
        "Assoc-acdm": "Associate degree, academic",
        "Assoc-voc": "Associate degree, vocational",
    },
    "marital-status": {
        "Married-civ-spouse": "Married, civilian spouse",
        "Never-married": "Never married",
        "Married-spouse-absent": "Married, spouse absent",
        "Married-AF-spouse": "Married, Armed Forces spouse",
    },
    "occupation": {
        "Tech-support": "Tech support",
        "Craft-repair": "Craft and repair",
        "Other-service": "Other service",
        "Exec-managerial": "Executive or managerial",
        "Prof-specialty": "Professional specialty",
        "Handlers-cleaners": "Handlers and cleaners",
        "Machine-op-inspct": "Machine operator or inspector",
        "Adm-clerical": "Administrative clerical",
        "Farming-fishing": "Farming or fishing",
        "Transport-moving": "Transport and moving",
        "Priv-house-serv": "Private household service",
        "Protective-serv": "Protective service",
        "Armed-Forces": "Armed Forces",
    },
    "relationship": {
        "Own-child": "Child",
        "Not-in-family": "Not in family",
        "Other-relative": "Other relative",
    },
    "race": {
        "Asian-Pac-Islander": "Asian or Pacific Islander",
        "Amer-Indian-Eskimo": "American Indian or Alaska Native",
        "Other": "Other / not specified",
    },
    "native-country": {
        "United-States": "United States",
        "Puerto-Rico": "Puerto Rico",
        "Outlying-US(Guam-USVI-etc)": "Outlying US territories",
        "Dominican-Republic": "Dominican Republic",
        "El-Salvador": "El Salvador",
        "Trinadad&Tobago": "Trinidad and Tobago",
        "Holand-Netherlands": "Holland / Netherlands",
    },
}
REVERSE_CATEGORY_DISPLAY_LABELS = {
    column: {label: value for value, label in labels.items()}
    for column, labels in CATEGORY_DISPLAY_LABELS.items()
}

DEFAULT_PROFILE = {
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

EXAMPLE_PROFILES: dict[str, dict[str, Any]] = {
    "Tech Worker": {
        "age": 34,
        "workclass": "Private",
        "fnlwgt": 180000,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "India",
    },
    "Blue Collar": {
        "age": 42,
        "workclass": "Private",
        "fnlwgt": 220000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    },
    "Recent Graduate": {
        "age": 23,
        "workclass": "Private",
        "fnlwgt": 145000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 35,
        "native-country": "United-States",
    },
    "Senior Executive": {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlwgt": 190000,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
    },
    "Part-time Worker": {
        "age": 29,
        "workclass": "Private",
        "fnlwgt": 165000,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 24,
        "native-country": "United-States",
    },
}

DRIFT_SCORES = pd.DataFrame(
    [
        {"feature": "sex", "score": 0.362, "method": "Jensen-Shannon"},
        {"feature": "race", "score": 0.231, "method": "Jensen-Shannon"},
        {"feature": "relationship", "score": 0.201, "method": "Jensen-Shannon"},
        {"feature": "hours-per-week", "score": 0.180, "method": "Wasserstein"},
        {"feature": "marital-status", "score": 0.127, "method": "Jensen-Shannon"},
        {"feature": "occupation", "score": 0.123, "method": "Jensen-Shannon"},
    ]
)

SUBGROUP_F1 = pd.DataFrame(
    [
        {"group": "Female", "f1": 0.6678},
        {"group": "Male", "f1": 0.7165},
        {"group": "White", "f1": 0.7140},
        {"group": "Black", "f1": 0.6526},
        {"group": "Asian-Pac-Islander", "f1": 0.6926},
        {"group": "Amer-Indian-Eskimo", "f1": 0.5000},
    ]
)

CI_STAGES = pd.DataFrame(
    [
        {"stage": "Lint", "duration": "6s", "seconds": 6},
        {"stage": "Unit Tests", "duration": "1m30s", "seconds": 90},
        {"stage": "Data Validation", "duration": "1m40s", "seconds": 100},
        {"stage": "Integration Tests", "duration": "10s", "seconds": 10},
        {"stage": "Coverage", "duration": "72%", "seconds": 12},
        {"stage": "Train", "duration": "2m58s", "seconds": 178},
        {"stage": "Model Validation", "duration": "1m13s", "seconds": 73},
    ]
)


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(show_spinner=False)
def github_actions_url() -> str:
    configured = os.environ.get("GITHUB_ACTIONS_URL")
    if configured:
        return configured

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        remote = result.stdout.strip()
    except Exception:
        return "https://github.com"

    repo = ""
    if remote.startswith("git@github.com:"):
        repo = remote.split(":", 1)[1]
    elif "github.com/" in remote:
        repo = remote.split("github.com/", 1)[1]

    repo = repo.removesuffix(".git").strip("/")
    if repo:
        return f"https://github.com/{repo}/actions"
    return "https://github.com"


def prefect_executable() -> str:
    local_prefect = PROJECT_ROOT / ".venv" / "bin" / "prefect"
    if local_prefect.exists():
        return str(local_prefect)
    return shutil.which("prefect") or "prefect"


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[Any | None, Any | None, str | None]:
    if not MODEL_PATH.exists():
        return None, None, f"Missing {safe_rel(MODEL_PATH)}"
    if not PIPELINE_PATH.exists():
        return None, None, f"Missing {safe_rel(PIPELINE_PATH)}"

    try:
        with PIPELINE_PATH.open("rb") as f:
            preprocessor = pickle.load(f)
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        return preprocessor, model, None
    except Exception as exc:
        logger.exception("Artifact loading failed")
        return None, None, f"Could not load model artifacts: {exc}"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        logger.exception("CSV loading failed for %s", path)
        return None, f"Could not load {safe_rel(path)}: {exc}"


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:
        logger.exception("Text loading failed for %s", path)
        return None, f"Could not load {safe_rel(path)}: {exc}"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{safe_rel(path)} not found"
    except Exception as exc:
        logger.exception("JSON loading failed for %s", path)
        return None, f"Could not load {safe_rel(path)}: {exc}"


def api_get(path: str, timeout: float = 1.5) -> tuple[Any | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json(), None
        return response.text, None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, payload: dict[str, Any], timeout: float = 3.0) -> tuple[Any | None, str | None]:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def get_api_status() -> dict[str, Any]:
    payload, error = api_get("/health")
    return {"up": error is None, "payload": payload, "error": error}


def get_mlflow_status() -> dict[str, Any]:
    if not MLFLOW_DB_PATH.exists():
        return {"up": False, "detail": "mlflow.db missing"}
    try:
        with sqlite3.connect(MLFLOW_DB_PATH) as conn:
            run_count = conn.execute("select count(*) from runs").fetchone()[0]
        return {"up": True, "detail": f"{run_count} run(s)", "uri": MLFLOW_TRACKING_URI}
    except Exception as exc:
        return {"up": False, "detail": str(exc), "uri": MLFLOW_TRACKING_URI}


def parse_prometheus_metrics(text: str | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "predictions_total": 0.0,
        "inference_latency_seconds": None,
        "model_version_info": "unknown",
    }
    if not text:
        return metrics

    latency_sum = None
    latency_count = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        value_match = re.search(r"\s(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$", line, re.IGNORECASE)
        if not value_match:
            continue
        value = float(value_match.group(1))
        if line.startswith("predictions_total"):
            metrics["predictions_total"] += value
        elif line.startswith("inference_latency_seconds_sum"):
            latency_sum = value
        elif line.startswith("inference_latency_seconds_count"):
            latency_count = value
        elif line.startswith("inference_latency_seconds "):
            metrics["inference_latency_seconds"] = value
        elif line.startswith("model_version_info"):
            metrics["model_version_info"] = value

    if latency_sum is not None and latency_count:
        metrics["inference_latency_seconds"] = latency_sum / latency_count
    return metrics


def validate_frame(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        errors.append("Missing columns: " + ", ".join(display_name(col) for col in missing))
        return errors

    for col in NUMERIC_FEATURES:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            errors.append(f"{display_name(col)} contains non-numeric or missing values")
        lo, hi = NUMERIC_RANGES[col]
        out_of_range = (coerced < lo) | (coerced > hi)
        if out_of_range.any():
            errors.append(f"{display_name(col)} has values outside {lo}-{hi}")

    for col in CATEGORICAL_FEATURES:
        unknown = ~df[col].isin(CATEGORY_OPTIONS[col])
        if unknown.any():
            errors.append(f"{display_name(col)} contains unknown categories")
    return errors


def display_name(column: str) -> str:
    return DISPLAY_LABELS.get(column, column)


def category_display_name(column: str, value: str) -> str:
    return CATEGORY_DISPLAY_LABELS.get(column, {}).get(value, value)


def display_selectbox(column: str) -> None:
    st.selectbox(
        display_name(column),
        CATEGORY_OPTIONS[column],
        key=f"input_{column}",
        format_func=lambda value: category_display_name(column, value),
    )


def display_feature_name(raw_name: str) -> str:
    name = raw_name
    if "__" in name:
        name = name.split("__", 1)[1]
    for column in sorted(FEATURE_ORDER, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            suffix = name[len(column):].lstrip("_")
            suffix_label = category_display_name(column, suffix) if suffix else suffix
            return f"{display_name(column)}: {suffix_label}" if suffix else display_name(column)
    return name.replace("_", " ").title()


def to_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in CATEGORICAL_FEATURES:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: category_display_name(column, value)
            )
    return display_df.rename(columns=DISPLAY_LABELS)


def normalize_uploaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=REVERSE_DISPLAY_LABELS).copy()
    for column in CATEGORICAL_FEATURES:
        if column in normalized.columns:
            reverse_values = REVERSE_CATEGORY_DISPLAY_LABELS.get(column, {})
            normalized[column] = normalized[column].map(
                lambda value: reverse_values.get(value, value)
            )
    return normalized


def normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df[FEATURE_ORDER].copy()
    for col in NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(int)
    return work


def predict_dataframe(df: pd.DataFrame, preprocessor: Any, model: Any) -> tuple[np.ndarray, np.ndarray | None]:
    X = preprocessor.transform(df[FEATURE_ORDER])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        positive = np.asarray(proba)[:, 1]
        threshold = get_decision_threshold(model)
        labels = np.where(positive >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
        return labels, positive
    preds = np.asarray(model.predict(X)).astype(int)
    labels = np.where(preds == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    return labels, None


def feature_importance_frame(preprocessor: Any, model: Any) -> pd.DataFrame:
    importance_model = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        importance_model = model.named_steps["model"]

    try:
        importances = np.asarray(importance_model.feature_importances_, dtype=float)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = [f"feature_{idx}" for idx in range(len(importances))]

    if len(names) != len(importances):
        names = [f"feature_{idx}" for idx in range(len(importances))]

    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .assign(feature=lambda frame: frame["feature"].map(display_feature_name))
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance")
    )


def apply_profile(profile_name: str) -> None:
    values = EXAMPLE_PROFILES.get(profile_name, DEFAULT_PROFILE)
    for key, value in values.items():
        st.session_state[f"input_{key}"] = value


def reset_profile() -> None:
    st.session_state["selected_profile"] = "Tech Worker"
    apply_profile("Tech Worker")


def result_badge(label: str) -> None:
    if label == POSITIVE_LABEL:
        color = "#f59e0b"
        background = "#fff7ed"
    else:
        color = "#15803d"
        background = "#f0fdf4"
    st.markdown(
        f"""
        <div style="border:1px solid {color}; background:{background}; color:{color};
                    border-radius:8px; padding:18px 20px; font-size:28px;
                    font-weight:700; text-align:center;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, ok: bool, detail: str = "") -> None:
    icon = "OK" if ok else "DOWN"
    color = "#15803d" if ok else "#b91c1c"
    st.markdown(
        f"""
        <div style="font-size:1rem;">
            <b>{label}:</b>
            <span style="color:{color}; font-weight:700;">{icon}</span>
            <span style="color:#475569;">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def architecture_figure() -> go.Figure:
    stages = [
        ("Data", "UCI Adult, DVC"),
        ("Training", "Prefect, MLflow, XGBoost"),
        ("Deployment", "FastAPI, Docker"),
        ("Monitoring", "Evidently, Prometheus"),
    ]
    fig = go.Figure()
    x_positions = [0.08, 0.36, 0.64, 0.92]
    for idx, ((title, subtitle), x) in enumerate(zip(stages, x_positions)):
        fig.add_shape(
            type="rect",
            x0=x - 0.105,
            x1=x + 0.105,
            y0=0.38,
            y1=0.70,
            line={"color": "#2563eb", "width": 2},
            fillcolor="#eff6ff",
            layer="below",
        )
        fig.add_annotation(x=x, y=0.59, text=f"<b>{title}</b>", showarrow=False, font={"size": 18})
        fig.add_annotation(x=x, y=0.48, text=subtitle, showarrow=False, font={"size": 12, "color": "#475569"})
        if idx < len(x_positions) - 1:
            fig.add_annotation(
                x=x_positions[idx + 1] - 0.13,
                y=0.54,
                ax=x + 0.13,
                ay=0.54,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#334155",
            )
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        plot_bgcolor="white",
    )
    return fig


def confidence_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "P(income > 50K)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 100], "color": "#ffedd5"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 3}, "value": 50},
            },
        )
    )
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 40, "b": 10})
    return fig


def render_sidebar(api_status: dict[str, Any], mlflow_status: dict[str, Any]) -> str:
    st.sidebar.title("Adult Income MLOps")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Pipeline Overview",
            "Single Prediction",
            "Batch Prediction",
            "MLflow Experiments",
            "Monitoring & Drift Detection",
            "Live API Monitor",
            "Model Card",
            "CI/CD Status",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Live System Status")
    st.sidebar.metric("API", "UP" if api_status["up"] else "DOWN")
    st.sidebar.caption(str(api_status["payload"] if api_status["up"] else api_status["error"]))
    st.sidebar.metric("MLflow", "UP" if mlflow_status["up"] else "DOWN")
    st.sidebar.caption(f"{MLFLOW_TRACKING_URI} - {mlflow_status['detail']}")
    st.sidebar.divider()
    st.sidebar.caption("Local predictions use models/best_model.pkl and data/processed/pipeline.pkl.")
    return page


def render_pipeline_overview(model_ready: bool, api_status: dict[str, Any], drift_alert: dict[str, Any] | None) -> None:
    st.title("Pipeline Overview")
    cols = st.columns(3)
    cols[0].metric("F1", f"{MODEL_METRICS['F1']:.4f}")
    cols[1].metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.4f}")
    cols[2].metric("ROC-AUC", f"{MODEL_METRICS['ROC-AUC']:.4f}")
    st.caption("Model metrics are a snapshot from the latest verified training run.")

    st.plotly_chart(architecture_figure(), use_container_width=True)

    st.subheader("Current System Status")
    c1, c2, c3 = st.columns(3)
    with c1:
        status_pill("Model loaded", model_ready, safe_rel(MODEL_PATH))
    with c2:
        detail = ""
        if api_status["up"] and isinstance(api_status["payload"], dict):
            detail = api_status["payload"].get("model_version", "")
        status_pill("FastAPI", api_status["up"], detail)
    with c3:
        action = (drift_alert or {}).get("action", "NO_ALERT")
        status_pill("Drift", action != "RETRAIN_REQUIRED", action)

    st.subheader("Pipeline Assets")
    asset_rows = [
        ("DVC params", PARAMS_PATH.exists(), safe_rel(PARAMS_PATH)),
        ("MLflow DB", MLFLOW_DB_PATH.exists(), MLFLOW_TRACKING_URI),
        ("Model artifact", MODEL_PATH.exists(), safe_rel(MODEL_PATH)),
        ("Preprocessing pipeline", PIPELINE_PATH.exists(), safe_rel(PIPELINE_PATH)),
        ("Evidently drift report", DRIFT_REPORT_PATH.exists(), safe_rel(DRIFT_REPORT_PATH)),
        ("Prometheus endpoint", api_status["up"], f"{API_BASE_URL}/metrics"),
    ]
    st.dataframe(
        pd.DataFrame(asset_rows, columns=["Component", "Available", "Location"]),
        use_container_width=True,
        hide_index=True,
    )


def render_single_prediction(preprocessor: Any, model: Any, model_ready: bool) -> None:
    st.title("Single Prediction")
    if "selected_profile" not in st.session_state:
        reset_profile()

    c_profile, c_reset = st.columns([3, 1])
    with c_profile:
        selected = st.selectbox(
            "Example profile",
            list(EXAMPLE_PROFILES.keys()),
            key="selected_profile",
            on_change=lambda: apply_profile(st.session_state["selected_profile"]),
        )
    with c_reset:
        st.write("")
        st.button("Reset", use_container_width=True, on_click=reset_profile)

    st.caption(f"Profile loaded: {selected}")
    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            for col in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss"]:
                lo, hi = NUMERIC_RANGES[col]
                help_text = (
                    "Census sampling weight: how many similar people this row represents."
                    if col == "fnlwgt"
                    else None
                )
                st.number_input(
                    display_name(col),
                    min_value=lo,
                    max_value=hi,
                    step=1,
                    key=f"input_{col}",
                    help=help_text,
                )
        with c2:
            st.number_input(
                display_name("hours-per-week"),
                min_value=NUMERIC_RANGES["hours-per-week"][0],
                max_value=NUMERIC_RANGES["hours-per-week"][1],
                step=1,
                key="input_hours-per-week",
            )
            for col in ["workclass", "education", "marital-status", "occupation"]:
                display_selectbox(col)
        with c3:
            for col in ["relationship", "race", "sex", "native-country"]:
                display_selectbox(col)
        submitted = st.form_submit_button("Predict", type="primary", disabled=not model_ready)

    if not model_ready:
        st.warning("Model artifacts are not loaded. Local predictions are unavailable.")
        return
    if not submitted:
        return

    record = {col: st.session_state[f"input_{col}"] for col in FEATURE_ORDER}
    frame = pd.DataFrame([record], columns=FEATURE_ORDER)
    errors = validate_frame(frame)
    if errors:
        st.error("Input validation failed.")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        labels, proba = predict_dataframe(frame, preprocessor, model)
    except Exception as exc:
        logger.exception("Single prediction failed")
        st.error(f"Prediction failed: {exc}")
        return

    label = str(labels[0])
    probability = float(proba[0]) if proba is not None else (1.0 if label == POSITIVE_LABEL else 0.0)
    r1, r2 = st.columns([1, 2])
    with r1:
        st.subheader("Prediction")
        result_badge(label)
    with r2:
        st.plotly_chart(confidence_gauge(probability), use_container_width=True)

    importance = feature_importance_frame(preprocessor, model)
    if importance.empty:
        st.info("Feature importance is not exposed by this model.")
    else:
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importances",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=430, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Input row sent to model"):
        st.dataframe(to_display_columns(frame).T.rename(columns={0: "value"}), use_container_width=True)


def render_batch_prediction(preprocessor: Any, model: Any, model_ready: bool) -> None:
    st.title("Batch Prediction")
    sample = to_display_columns(pd.DataFrame([EXAMPLE_PROFILES[name] for name in EXAMPLE_PROFILES]))
    st.download_button(
        "Download CSV Template",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="adult_income_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with the 14 Adult Income feature columns.")
        return
    if not model_ready:
        st.warning("Model artifacts are not loaded. Batch predictions are unavailable.")
        return

    try:
        df = normalize_uploaded_columns(pd.read_csv(io.BytesIO(uploaded.getvalue())))
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    errors = validate_frame(df)
    if errors:
        st.error("CSV validation failed.")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        work = normalize_input_frame(df)
        labels, proba = predict_dataframe(work, preprocessor, model)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        st.error(f"Batch prediction failed: {exc}")
        return

    results = work.copy()
    results["prediction"] = labels
    if proba is not None:
        results["prob_>50K"] = np.round(proba, 4)

    st.subheader("Summary Statistics")
    total = len(results)
    positive_count = int((results["prediction"] == POSITIVE_LABEL).sum())
    negative_count = total - positive_count
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{total:,}")
    c2.metric(">50K", f"{positive_count:,}")
    c3.metric("<=50K", f"{negative_count:,}")

    dist = results["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
    fig = px.pie(dist, names="prediction", values="count", title="Prediction Distribution", hole=0.35)
    fig.update_traces(marker={"colors": ["#16a34a", "#f59e0b"]})
    st.plotly_chart(fig, use_container_width=True)

    display_results = to_display_columns(results)

    def color_display_prediction(row: pd.Series) -> list[str]:
        color = "background-color: #ffedd5" if row["prediction"] == POSITIVE_LABEL else "background-color: #dcfce7"
        return [color if col == "prediction" else "" for col in row.index]

    st.dataframe(display_results.style.apply(color_display_prediction, axis=1), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Results CSV",
        data=display_results.to_csv(index=False).encode("utf-8"),
        file_name="adult_income_predictions.csv",
        mime="text/csv",
    )


def render_mlflow_experiments() -> None:
    st.title("MLflow Experiments")
    df, error = load_csv(EXPERIMENT_LOG_PATH)
    if error or df is None:
        st.warning(error)
        return

    metric_cols = {
        "metrics.f1": "F1",
        "metrics.accuracy": "Accuracy",
        "metrics.precision": "Precision",
        "metrics.recall": "Recall",
        "metrics.roc_auc": "ROC-AUC",
    }
    plot_df = df[["params.model_type", *metric_cols.keys()]].rename(columns={"params.model_type": "Model", **metric_cols})
    long_df = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(
        long_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Comparison Across Metrics",
        text_auto=".3f",
    )
    fig.update_layout(yaxis_range=[0, 1], height=520)
    st.plotly_chart(fig, use_container_width=True)

    winner_idx = df["metrics.f1"].astype(float).idxmax()
    winner = df.loc[winner_idx]
    display = plot_df.copy()
    display["Winner"] = display["Model"].eq(winner["params.model_type"])

    def highlight_winner(row: pd.Series) -> list[str]:
        return ["background-color: #dcfce7; font-weight: 700" if row["Winner"] else "" for _ in row]

    st.subheader("Experiment Runs")
    st.dataframe(display.style.apply(highlight_winner, axis=1), use_container_width=True, hide_index=True)

    st.subheader("Best Hyperparameters")
    param_cols = [col for col in df.columns if col.startswith("params.") and pd.notna(winner[col]) and str(winner[col]) != ""]
    params = {col.replace("params.", ""): winner[col] for col in param_cols}
    st.json(params)


def render_monitoring() -> None:
    st.title("Monitoring & Drift Detection")
    alert, alert_error = load_json(DRIFT_ALERT_PATH)
    if alert and alert.get("action") == "RETRAIN_REQUIRED":
        st.error("RETRAIN_REQUIRED: production drift exceeds the configured threshold.")
        retraining = alert.get("retraining", {})
        command = retraining.get(
            "manual_trigger_command",
            'prefect deployment run "adult_income_training_pipeline/adult-income-weekly"',
        )
        st.caption("Retraining is connected to the Prefect deployment and can be triggered manually.")
        st.code(command, language="bash")
        if st.button("Trigger Prefect Retraining", type="primary"):
            try:
                completed = subprocess.run(
                    [
                        prefect_executable(),
                        "deployment",
                        "run",
                        retraining.get(
                            "deployment",
                            "adult_income_training_pipeline/adult-income-weekly",
                        ),
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if completed.returncode == 0:
                    st.success("Prefect retraining run submitted.")
                else:
                    st.warning(f"Prefect trigger exited with code {completed.returncode}.")
                with st.expander("Prefect trigger output"):
                    st.code((completed.stdout + "\n" + completed.stderr).strip() or "(no output)")
            except Exception as exc:
                logger.exception("Prefect retraining trigger failed")
                st.warning(f"Could not trigger Prefect retraining: {exc}")
    elif alert_error:
        st.warning(alert_error)
    else:
        st.success("No retraining alert found.")

    c1, c2 = st.columns(2)
    c1.metric("Baseline Drift Ratio", "0.00%")
    c2.metric("Production Drift Ratio", "37.50%")
    st.caption("Drift values are a snapshot from the latest generated Evidently reports.")
    summary = pd.DataFrame(
        [
            {"dataset": "baseline", "drift_ratio": "0.00%", "drifted_features": 0},
            {"dataset": "production", "drift_ratio": "37.50%", "drifted_features": 6},
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        DRIFT_SCORES.assign(feature=lambda frame: frame["feature"].map(display_name)).sort_values("score"),
        x="score",
        y="feature",
        color="method",
        orientation="h",
        title="Drift Scores for Drifted Features",
        text="score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, xaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Run Monitoring Now", type="primary"):
        try:
            completed = subprocess.run(
                [sys.executable, "monitoring/run_monitoring.py"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if completed.returncode == 0:
                st.success("Monitoring completed successfully.")
                load_json.clear()
                load_text.clear()
            else:
                st.warning(f"Monitoring exited with code {completed.returncode}.")
            with st.expander("Monitoring output"):
                st.code((completed.stdout + "\n" + completed.stderr).strip() or "(no output)")
        except Exception as exc:
            logger.exception("Monitoring subprocess failed")
            st.warning(f"Could not run monitoring: {exc}")

    st.subheader("Evidently Drift Report")
    html, error = load_text(DRIFT_REPORT_PATH)
    if html:
        components.html(html, height=700, scrolling=True)
    else:
        st.warning(error)


def render_live_api_monitor() -> None:
    st.title("Live API Monitor")
    health, health_error = api_get("/health")
    if health_error:
        st.error(f"FastAPI is down: {health_error}")
    else:
        st.success("FastAPI is healthy.")
        st.json(health)

    metrics_text, metrics_error = api_get("/metrics")
    parsed = parse_prometheus_metrics(metrics_text if not metrics_error else None)
    c1, c2, c3 = st.columns(3)
    c1.metric("predictions_total", f"{parsed['predictions_total']:.0f}")
    latency = parsed["inference_latency_seconds"]
    c2.metric("inference_latency_seconds", "n/a" if latency is None else f"{latency:.4f}")
    c3.metric("model_version_info", str(parsed["model_version_info"]))
    if metrics_error:
        st.warning(f"Could not scrape Prometheus metrics: {metrics_error}")

    if st.button("Send Test Request", type="primary"):
        response, error = api_post("/predict", EXAMPLE_PROFILES["Tech Worker"])
        if error:
            st.warning(f"Test request failed: {error}")
        else:
            st.success("Test request succeeded.")
            st.json(response)


def render_model_card() -> None:
    st.title("Model Card")
    markdown, error = load_text(MODEL_CARD_PATH)
    if markdown:
        st.markdown(markdown)
    else:
        st.warning(error)

    fig = px.bar(
        SUBGROUP_F1.sort_values("f1"),
        x="f1",
        y="group",
        orientation="h",
        title="Subgroup F1 Scores",
        text="f1",
        color="f1",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=430, xaxis_range=[0, 0.8], coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Limitations and Ethical Considerations")
    st.warning(
        "This educational model is trained on 1994 Census data and encodes historical "
        "income disparities across demographic groups. It must not be used for hiring, "
        "lending, benefits, credit, or other consequential decisions."
    )


def render_cicd_status() -> None:
    st.title("CI/CD Status")
    st.caption("Snapshot from the latest verified local/CI run. Open GitHub Actions for the live workflow state.")
    st.subheader("Pipeline")
    fig = go.Figure()
    for idx, row in CI_STAGES.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[idx],
                y=[1],
                mode="markers+text",
                marker={"size": 54, "color": "#16a34a", "symbol": "circle"},
                text=f"{row['stage']}<br>{row['duration']}",
                textposition="bottom center",
                name=row["stage"],
                showlegend=False,
            )
        )
        if idx < len(CI_STAGES) - 1:
            fig.add_shape(
                type="line",
                x0=idx + 0.18,
                x1=idx + 0.82,
                y0=1,
                y1=1,
                line={"color": "#16a34a", "width": 4},
            )
    fig.update_layout(
        height=300,
        xaxis={"visible": False, "range": [-0.6, len(CI_STAGES) - 0.4]},
        yaxis={"visible": False, "range": [0.55, 1.35]},
        margin={"l": 20, "r": 20, "t": 20, "b": 80},
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Coverage", "72%")
    c2.metric("serving/app.py", "69%")
    c3.metric("feature_engineering.py", "95%")

    coverage = pd.DataFrame(
        [
            {"file": "feature_engineering.py", "coverage": 95},
            {"file": "evaluate.py", "coverage": 88},
            {"file": "serving/app.py", "coverage": 69},
            {"file": "src total", "coverage": 72},
        ]
    )
    fig_cov = px.bar(coverage, x="file", y="coverage", text="coverage", title="Coverage by File")
    fig_cov.update_traces(texttemplate="%{text}%", marker_color="#2563eb")
    fig_cov.update_layout(yaxis_range=[0, 100], height=360)
    st.plotly_chart(fig_cov, use_container_width=True)

    st.link_button("Open GitHub Actions", github_actions_url())


def main() -> None:
    st.set_page_config(
        page_title="Adult Income MLOps Demo",
        page_icon="🤖",
        layout="wide",
    )

    preprocessor, model, load_error = load_artifacts()
    model_ready = preprocessor is not None and model is not None and load_error is None
    api_status = get_api_status()
    mlflow_status = get_mlflow_status()
    drift_alert, _ = load_json(DRIFT_ALERT_PATH)

    page = render_sidebar(api_status, mlflow_status)
    if load_error:
        st.warning(load_error)

    try:
        if page == "Pipeline Overview":
            render_pipeline_overview(model_ready, api_status, drift_alert)
        elif page == "Single Prediction":
            render_single_prediction(preprocessor, model, model_ready)
        elif page == "Batch Prediction":
            render_batch_prediction(preprocessor, model, model_ready)
        elif page == "MLflow Experiments":
            render_mlflow_experiments()
        elif page == "Monitoring & Drift Detection":
            render_monitoring()
        elif page == "Live API Monitor":
            render_live_api_monitor()
        elif page == "Model Card":
            render_model_card()
        elif page == "CI/CD Status":
            render_cicd_status()
    except Exception as exc:
        logger.exception("Page render failed")
        st.error(f"This page encountered an unexpected error: {exc}")


if __name__ == "__main__":
    main()
