"""
Integration tests for the FastAPI serving application.
Uses TestClient with mocked model and preprocessor so no MLflow server needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from unittest import mock
from fastapi.testclient import TestClient


VALID_RECORD = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

BATCH_REQUEST = {"records": [VALID_RECORD, VALID_RECORD]}


def _mock_preprocessor():
    m = mock.MagicMock()
    m.transform.return_value = np.zeros((1, 105))
    return m


def _mock_model(prob=0.72):
    m = mock.MagicMock()
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    m.decision_threshold_ = 0.5
    return m


@pytest.fixture
def client():
    """TestClient with model and preprocessor mocked out."""
    with mock.patch("src.serving.app.load_model"):
        import src.serving.app as app_module

        app_module.model = _mock_model()
        app_module.preprocessor = _mock_preprocessor()
        app_module.model_info = {
            "name": "adult_income_classifier",
            "version": "1",
            "stage": "Production",
        }
        yield TestClient(app_module.app)


# ------------------------------------------------------------------ #
# /health
# ------------------------------------------------------------------ #
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_required_fields(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model_name" in data
    assert "model_version" in data


def test_health_status_is_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_returns_503_when_model_not_loaded(client):
    import src.serving.app as app_module

    app_module.model = None
    app_module.preprocessor = None
    app_module.load_error = "missing artifacts"

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "model_not_loaded"


# ------------------------------------------------------------------ #
# /predict
# ------------------------------------------------------------------ #
def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200


def test_predict_response_schema(client):
    data = client.post("/predict", json=VALID_RECORD).json()
    assert "prediction" in data
    assert "confidence" in data
    assert "decision_threshold" in data
    assert "model_version" in data


def test_predict_prediction_values(client):
    data = client.post("/predict", json=VALID_RECORD).json()
    assert data["prediction"] in [">50K", "<=50K"]


def test_predict_confidence_range(client):
    data = client.post("/predict", json=VALID_RECORD).json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_uses_model_decision_threshold(client):
    import src.serving.app as app_module

    app_module.model = _mock_model(prob=0.72)
    app_module.model.decision_threshold_ = 0.8

    data = client.post("/predict", json=VALID_RECORD).json()

    assert data["prediction"] == "<=50K"
    assert data["decision_threshold"] == 0.8


def test_predict_invalid_input_returns_422(client):
    response = client.post("/predict", json={"invalid_field": "garbage"})
    assert response.status_code == 422


def test_predict_age_out_of_range_returns_422(client):
    bad = {**VALID_RECORD, "age": 200}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workclass", "Gig-economy"),
        ("education", "Bootcamp"),
        ("marital-status", "Complicated"),
        ("occupation", "Influencer"),
        ("relationship", "Roommate"),
        ("race", "Alien"),
        ("sex", "Dragon"),
        ("native-country", "Atlantis"),
    ],
)
def test_predict_invalid_category_returns_422(client, field, value):
    bad = {**VALID_RECORD, field: value}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ------------------------------------------------------------------ #
# /predict/batch
# ------------------------------------------------------------------ #
def test_batch_predict_returns_200(client):
    response = client.post("/predict/batch", json=BATCH_REQUEST)
    assert response.status_code == 200


def test_batch_predict_returns_correct_count(client):
    data = client.post("/predict/batch", json=BATCH_REQUEST).json()
    assert len(data["predictions"]) == 2


def test_batch_predict_empty_records_returns_422(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422


def test_batch_predict_invalid_category_returns_422(client):
    bad_record = {**VALID_RECORD, "race": "Alien"}
    response = client.post("/predict/batch", json={"records": [VALID_RECORD, bad_record]})
    assert response.status_code == 422
