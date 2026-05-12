"""
Model validation script used in the CI/CD pipeline.
Loads the saved best model and asserts it meets the minimum F1 threshold.
Does NOT require an MLflow server — loads model from disk.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yaml
from sklearn.metrics import f1_score

from src.prediction import predict_positive_class


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate() -> None:
    params = load_params()

    model_path = params["preprocessing"]["best_model_path"]
    threshold = params["training"]["min_f1_threshold"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv("data/splits/X_test.csv").values
    y_test = pd.read_csv("data/splits/y_test.csv").values.ravel()

    y_pred = predict_positive_class(model, X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"Model F1 on test set : {f1:.4f}")
    print(f"Minimum threshold    : {threshold}")

    if f1 < threshold:
        print(f"FAIL: F1 {f1:.4f} is below threshold {threshold}")
        sys.exit(1)

    print("PASS: Model meets performance threshold.")


if __name__ == "__main__":
    evaluate()
