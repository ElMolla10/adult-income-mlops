"""
Focused tests for pipeline scripts that are otherwise exercised mostly through
DVC/CI. These keep whole-src coverage meaningful without requiring live MLflow.
"""
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data import load_data
from src.evaluation import evaluate as evaluate_module
from src.features import feature_engineering
from src.training import train as train_module
from monitoring import run_monitoring


COLUMNS = [
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
    "income",
]


def _adult_row(age=39, race="White", sex="Male", income="<=50K"):
    return [
        age,
        "Private",
        77516,
        "Bachelors",
        13,
        "Never-married",
        "Adm-clerical",
        "Not-in-family",
        race,
        sex,
        0,
        0,
        40,
        "United-States",
        income,
    ]


def _write_csv_rows(path: Path, rows, header=False):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False, header=header)


def test_load_adult_data_strips_test_label_period(tmp_path):
    train_path = tmp_path / "adult.data"
    test_path = tmp_path / "adult.test"
    _write_csv_rows(train_path, [_adult_row(income="<=50K")])
    test_path.write_text("| junk header\n", encoding="utf-8")
    pd.DataFrame([_adult_row(income=">50K.")], columns=COLUMNS).to_csv(
        test_path, index=False, header=False, mode="a"
    )

    params = {
        "data": {
            "columns": COLUMNS,
            "raw_train": str(train_path),
            "raw_test": str(test_path),
            "target_column": "income",
        }
    }

    train_df, test_df = load_data.load_adult_data(params)

    assert train_df.loc[0, "income"] == "<=50K"
    assert test_df.loc[0, "income"] == ">50K"


def test_prepare_data_writes_processed_files(tmp_path, monkeypatch):
    processed_train = Path("data/processed/train.csv")
    processed_test = Path("data/processed/test.csv")
    params = {
        "data": {
            "processed_train": str(processed_train),
            "processed_test": str(processed_test),
        }
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(load_data, "load_params", lambda: params)
    monkeypatch.setattr(
        load_data,
        "load_adult_data",
        lambda _: (
            pd.DataFrame([_adult_row()], columns=COLUMNS),
            pd.DataFrame([_adult_row(income=">50K")], columns=COLUMNS),
        ),
    )

    load_data.prepare_data()

    assert processed_train.exists()
    assert processed_test.exists()


class StaticModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return np.repeat(self.prediction, len(X))


def test_evaluate_passes_when_f1_meets_threshold(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(StaticModel(1), f)
    split_dir = tmp_path / "data" / "splits"
    split_dir.mkdir(parents=True)
    pd.DataFrame([[0], [1]]).to_csv(split_dir / "X_test.csv", index=False)
    pd.Series([1, 1], name="income").to_csv(split_dir / "y_test.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        evaluate_module,
        "load_params",
        lambda: {
            "preprocessing": {"best_model_path": str(model_path)},
            "training": {"min_f1_threshold": 0.9},
        },
    )

    evaluate_module.evaluate()


def test_evaluate_exits_when_f1_below_threshold(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(StaticModel(0), f)
    split_dir = tmp_path / "data" / "splits"
    split_dir.mkdir(parents=True)
    pd.DataFrame([[0], [1]]).to_csv(split_dir / "X_test.csv", index=False)
    pd.Series([1, 1], name="income").to_csv(split_dir / "y_test.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        evaluate_module,
        "load_params",
        lambda: {
            "preprocessing": {"best_model_path": str(model_path)},
            "training": {"min_f1_threshold": 0.9},
        },
    )

    with pytest.raises(SystemExit):
        evaluate_module.evaluate()


def test_feature_engineering_writes_preprocessor_and_splits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        [
            _adult_row(age=30, race="White", sex="Male", income="<=50K"),
            _adult_row(age=45, race="Black", sex="Female", income=">50K"),
            _adult_row(age=50, race="White", sex="Male", income=">50K"),
        ],
        columns=COLUMNS,
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [
            _adult_row(age=31, race="White", sex="Male", income="<=50K"),
            _adult_row(age=46, race="Black", sex="Female", income=">50K"),
        ],
        columns=COLUMNS,
    ).to_csv(test_path, index=False)

    monkeypatch.setattr(
        feature_engineering,
        "load_params",
        lambda: {
            "data": {
                "processed_train": str(train_path),
                "processed_test": str(test_path),
                "target_column": "income",
                "positive_label": ">50K",
                "reference_data": "data/splits/reference.csv",
                "production_data": "data/splits/production.csv",
            },
            "preprocessing": {
                "numeric_features": [
                    "age",
                    "fnlwgt",
                    "education-num",
                    "capital-gain",
                    "capital-loss",
                    "hours-per-week",
                ],
                "categorical_features": [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                ],
                "pipeline_path": "data/processed/pipeline.pkl",
            },
        },
    )

    feature_engineering.featurize()

    assert Path("data/processed/pipeline.pkl").exists()
    assert Path("data/splits/X_train.csv").exists()
    assert Path("data/splits/production.csv").exists()
    assert len(pd.read_csv("data/splits/production.csv")) == 1


def test_training_run_experiment_uses_smote_inside_cv(monkeypatch):
    captured = {}

    class DummyModel:
        def predict(self, X):
            return np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    class FakeSearch:
        def __init__(self, estimator, param_distributions, **kwargs):
            captured["estimator"] = estimator
            captured["param_distributions"] = param_distributions
            self.best_estimator_ = DummyModel()
            self.best_params_ = {"model__C": 1.0}

        def fit(self, X, y):
            captured["fit_rows"] = len(X)

    class FakeRun:
        info = SimpleNamespace(run_id="run-1")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(train_module, "RandomizedSearchCV", FakeSearch)
    monkeypatch.setattr(train_module.mlflow, "start_run", lambda **_: FakeRun())
    monkeypatch.setattr(train_module.mlflow, "log_param", lambda *_, **__: None)
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda params: captured.setdefault("logged_params", params))
    monkeypatch.setattr(train_module.mlflow, "log_metrics", lambda *_, **__: None)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", lambda *_, **__: None)

    f1, run_id, _ = train_module.run_experiment(
        "LogisticRegression",
        mock.MagicMock(),
        {"C": [1.0]},
        np.array([[0], [1]]),
        np.array([[0], [1]]),
        np.array([0, 1]),
        np.array([0, 1]),
        {
            "preprocessing": {"random_state": 42},
            "training": {"hpo_n_iter": 1, "hpo_cv": 2, "metric": "f1"},
            "serving": {"model_name": "adult_income_classifier"},
        },
        "exp-1",
    )

    assert run_id == "run-1"
    assert f1 == 1.0
    assert "smote" in captured["estimator"].named_steps
    assert captured["param_distributions"] == {"model__C": [1.0]}
    assert captured["logged_params"] == {"C": 1.0}


def test_drift_alert_includes_prefect_retraining_command():
    alert = run_monitoring.build_retraining_alert(
        drift_ratio=0.375,
        threshold=0.2,
        drifted_features=["race", "sex"],
    )

    assert alert["action"] == "RETRAIN_REQUIRED"
    assert alert["retraining"]["orchestrator"] == "prefect"
    assert alert["retraining"]["deployment"] == "adult_income_training_pipeline/adult-income-weekly"
    assert (
        alert["retraining"]["manual_trigger_command"]
        == 'prefect deployment run "adult_income_training_pipeline/adult-income-weekly"'
    )
    assert alert["retraining"]["local_fallback_command"] == "python orchestration/flows/training_flow.py"
