"""
Model serving application.
Endpoints:
  GET  /health          — returns model name/version/status
  POST /predict         — single prediction with confidence
  POST /predict/batch   — batch predictions
  GET  /metrics         — Prometheus metrics scrape endpoint
"""
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src.prediction import get_decision_threshold


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


params = load_params()

# ------------------------------------------------------------------ #
# Prometheus metrics
# ------------------------------------------------------------------ #
PREDICTION_COUNTER = Counter(
    "predictions_total", "Inference count by predicted class", ["predicted_class"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "End-to-end inference latency"
)
CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
AGE_HISTOGRAM = Histogram(
    "input_feature_age",
    "Age distribution of incoming requests",
    buckets=[20, 30, 40, 50, 60, 70, 90],
)
HOURS_HISTOGRAM = Histogram(
    "input_feature_hours_per_week",
    "Hours-per-week distribution of incoming requests",
    buckets=[10, 20, 30, 40, 50, 60, 80],
)
MODEL_VERSION_GAUGE = Gauge("model_version_info", "Current model version number")

# ------------------------------------------------------------------ #
# Global state
# ------------------------------------------------------------------ #
model = None
preprocessor = None
model_info: dict = {
    "name": params["serving"]["model_name"],
    "version": "unknown",
    "stage": params["serving"]["model_stage"],
}
load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    result = load_model()
    if hasattr(result, "__await__"):
        await result
    yield


app = FastAPI(title="Adult Income Classifier", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #
Workclass = Literal[
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]
Education = Literal[
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
]
MaritalStatus = Literal[
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]
Occupation = Literal[
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
]
Relationship = Literal[
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]
Race = Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
Sex = Literal["Female", "Male"]
NativeCountry = Literal[
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
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., ge=17, le=100)
    workclass: Optional[Workclass] = None
    fnlwgt: int = Field(..., gt=0)
    education: Education
    education_num: int = Field(..., alias="education-num", ge=1, le=16)
    marital_status: MaritalStatus = Field(..., alias="marital-status")
    occupation: Optional[Occupation] = None
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int = Field(..., alias="capital-gain", ge=0)
    capital_loss: int = Field(..., alias="capital-loss", ge=0)
    hours_per_week: int = Field(..., alias="hours-per-week", ge=1, le=99)
    native_country: Optional[NativeCountry] = Field(None, alias="native-country")


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: str
    confidence: float
    decision_threshold: float
    model_version: str


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def record_to_df(record: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": record.age,
                "workclass": record.workclass,
                "fnlwgt": record.fnlwgt,
                "education": record.education,
                "education-num": record.education_num,
                "marital-status": record.marital_status,
                "occupation": record.occupation,
                "relationship": record.relationship,
                "race": record.race,
                "sex": record.sex,
                "capital-gain": record.capital_gain,
                "capital-loss": record.capital_loss,
                "hours-per-week": record.hours_per_week,
                "native-country": record.native_country,
            }
        ]
    )


# ------------------------------------------------------------------ #
# Startup
# ------------------------------------------------------------------ #
async def load_model() -> None:
    global model, preprocessor, load_error
    load_error = None
    model_loaded = False

    def load_local_model() -> None:
        global model
        model_path = params["preprocessing"]["best_model_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_info["version"] = "local"

    def load_mlflow_model() -> None:
        global model
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", params["training"]["mlflow_tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = (
            f"models:/{params['serving']['model_name']}"
            f"/{params['serving']['model_stage']}"
        )
        model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            params["serving"]["model_name"],
            stages=[params["serving"]["model_stage"]],
        )
        if versions:
            model_info["version"] = versions[0].version
            MODEL_VERSION_GAUGE.set(float(versions[0].version))

    use_mlflow = os.environ.get("ADULT_INCOME_USE_MLFLOW", "").lower() in {
        "1",
        "true",
        "yes",
    }
    primary_loader, fallback_loader = (
        (load_mlflow_model, load_local_model) if use_mlflow else (load_local_model, load_mlflow_model)
    )

    try:
        primary_loader()
        model_loaded = True
    except Exception as primary_exc:
        try:
            fallback_loader()
            model_loaded = True
        except Exception as fallback_exc:
            load_error = (
                f"Primary model load failed: {primary_exc}. "
                f"Fallback model load failed: {fallback_exc}."
            )

    try:
        pipeline_path = params["preprocessing"]["pipeline_path"]
        with open(pipeline_path, "rb") as f:
            preprocessor = pickle.load(f)
    except Exception as pipeline_exc:
        load_error = f"{load_error or ''} Preprocessor load failed: {pipeline_exc}."

    if model_loaded and preprocessor is not None and load_error is None:
        print(f"Model loaded: {model_info['name']} v{model_info['version']}")
    else:
        print(f"Model failed to load: {load_error or 'unknown error'}")


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
def health():
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_loaded",
                "model_name": model_info["name"],
                "model_version": model_info["version"],
                "model_stage": model_info["stage"],
                "error": load_error,
            },
        )
    return {
        "status": "ok",
        "model_name": model_info["name"],
        "model_version": model_info["version"],
        "model_stage": model_info["stage"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.time()
    df = record_to_df(request)
    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    threshold = get_decision_threshold(model)
    pred = ">50K" if prob >= threshold else "<=50K"
    latency = time.time() - t0

    INFERENCE_LATENCY.observe(latency)
    PREDICTION_COUNTER.labels(predicted_class=pred).inc()
    CONFIDENCE_HISTOGRAM.observe(prob)
    AGE_HISTOGRAM.observe(request.age)
    HOURS_HISTOGRAM.observe(request.hours_per_week)

    return PredictResponse(
        prediction=pred,
        confidence=round(prob, 4),
        decision_threshold=round(threshold, 4),
        model_version=str(model_info["version"]),
    )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    return {"predictions": [predict(r) for r in request.records]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
