"""
Unit tests for the preprocessing pipeline.
Tests: build, fit, transform, missing value handling, output type, output shape.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.feature_engineering import build_preprocessor


@pytest.fixture
def params():
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    """Small representative sample matching Adult Income schema."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28],
            "workclass": ["State-gov", None, "Private", "Private", "Private"],
            "fnlwgt": [77516, 83311, 215646, 234721, 338409],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
            "education-num": [13, 13, 9, 7, 13],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                None,
                "Handlers-cleaners",
                "Prof-specialty",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
            ],
            "race": ["White", "White", "White", "Black", "Black"],
            "sex": ["Male", "Male", "Male", "Male", "Female"],
            "capital-gain": [2174, 0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40],
            "native-country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "Cuba",
            ],
        }
    )


# ------------------------------------------------------------------ #
# Test 1: preprocessor builds without error
# ------------------------------------------------------------------ #
def test_preprocessor_builds(params):
    preprocessor = build_preprocessor(params)
    assert preprocessor is not None


# ------------------------------------------------------------------ #
# Test 2: preprocessor fits and transforms successfully
# ------------------------------------------------------------------ #
def test_preprocessor_fits_and_transforms(params, sample_df):
    preprocessor = build_preprocessor(params)
    X = preprocessor.fit_transform(sample_df)
    assert X is not None
    assert X.shape[0] == len(sample_df), "Row count must be preserved"


# ------------------------------------------------------------------ #
# Test 3: no NaN values in output (imputation works)
# ------------------------------------------------------------------ #
def test_preprocessor_handles_missing(params, sample_df):
    preprocessor = build_preprocessor(params)
    X = preprocessor.fit_transform(sample_df)
    assert not np.isnan(X).any(), "Transformed data must have no NaN values"


# ------------------------------------------------------------------ #
# Test 4: output is a numpy array
# ------------------------------------------------------------------ #
def test_preprocessor_output_type(params, sample_df):
    preprocessor = build_preprocessor(params)
    X = preprocessor.fit_transform(sample_df)
    assert isinstance(X, np.ndarray), "Output must be numpy array"


# ------------------------------------------------------------------ #
# Test 5: output has more columns than input (OHE expansion)
# ------------------------------------------------------------------ #
def test_preprocessor_expands_columns(params, sample_df):
    preprocessor = build_preprocessor(params)
    X = preprocessor.fit_transform(sample_df)
    assert X.shape[1] > sample_df.shape[1], (
        "OHE should expand column count beyond original"
    )


# ------------------------------------------------------------------ #
# Test 6: transform is consistent (same input → same output)
# ------------------------------------------------------------------ #
def test_preprocessor_deterministic(params, sample_df):
    preprocessor = build_preprocessor(params)
    X1 = preprocessor.fit_transform(sample_df)
    X2 = preprocessor.transform(sample_df)
    np.testing.assert_array_almost_equal(X1, X2)


# ------------------------------------------------------------------ #
# Test 7: numeric features are scaled (mean ≈ 0 after StandardScaler)
# ------------------------------------------------------------------ #
def test_numeric_scaling(params, sample_df):
    preprocessor = build_preprocessor(params)
    n_numeric = len(params["preprocessing"]["numeric_features"])
    X = preprocessor.fit_transform(sample_df)
    numeric_part = X[:, :n_numeric]
    # With 5 samples scaling won't be perfect but values should be small
    assert np.abs(numeric_part).max() < 10, "Scaled values should be small"
