"""
Stage 2: preprocess
Builds and fits a scikit-learn Pipeline, transforms data, saves all artifacts.
Also creates reference / production splits for drift monitoring.
"""
import os
import pickle
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.getcwd())

from src.features.transformers import NoneToNaNTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_preprocessor(params: dict) -> ColumnTransformer:
    """Build a ColumnTransformer with numeric and categorical sub-pipelines."""
    numeric_features = params["preprocessing"]["numeric_features"]
    categorical_features = params["preprocessing"]["categorical_features"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("none_to_nan", NoneToNaNTransformer()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def featurize() -> None:
    params = load_params()
    target = params["data"]["target_column"]
    positive_label = params["data"]["positive_label"]

    train_df = pd.read_csv(params["data"]["processed_train"])
    test_df = pd.read_csv(params["data"]["processed_test"])

    X_train = train_df.drop(columns=[target])
    y_train = (train_df[target] == positive_label).astype(int)

    X_test = test_df.drop(columns=[target])
    y_test = (test_df[target] == positive_label).astype(int)

    preprocessor = build_preprocessor(params)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("Class distribution before model-time SMOTE:")
    print(y_train.value_counts().sort_index().to_string())

    # Save fitted preprocessor pipeline
    os.makedirs("data/processed", exist_ok=True)
    pipeline_path = params["preprocessing"]["pipeline_path"]
    with open(pipeline_path, "wb") as f:
        pickle.dump(preprocessor, f)

    # Save train/test splits as CSVs
    os.makedirs("data/splits", exist_ok=True)
    pd.DataFrame(X_train_t).to_csv("data/splits/X_train.csv", index=False)
    pd.DataFrame(X_test_t).to_csv("data/splits/X_test.csv", index=False)
    y_train.to_csv("data/splits/y_train.csv", index=False)
    y_test.to_csv("data/splits/y_test.csv", index=False)

    # ------------------------------------------------------------------ #
    # Drift simulation
    # Reference = full training set (raw, for Evidently)
    # Production = test set filtered to Male + White (subgroup shift)
    # This creates measurable distribution shift on sex & race features
    # ------------------------------------------------------------------ #
    train_df.to_csv(params["data"]["reference_data"], index=False)

    production_df = test_df[
        (test_df["sex"] == "Male") & (test_df["race"] == "White")
    ].copy()
    production_df.to_csv(params["data"]["production_data"], index=False)

    print(f"Train original : {X_train_t.shape}")
    print(f"Test           : {X_test_t.shape}")
    print(f"Reference rows: {train_df.shape[0]}")
    print(f"Production rows (drifted): {production_df.shape[0]}")
    print(f"Pipeline saved to {pipeline_path}")
    print("Stage 2 (preprocess) complete.")


if __name__ == "__main__":
    featurize()
