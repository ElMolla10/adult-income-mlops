"""
Tests for the Pandera data validation schema.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import pytest
from src.data.validate_data import validate_data


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "age": [39, 50],
            "workclass": ["State-gov", "Private"],
            "fnlwgt": [77516, 83311],
            "education": ["Bachelors", "HS-grad"],
            "education-num": [13, 9],
            "marital-status": ["Never-married", "Divorced"],
            "occupation": ["Adm-clerical", "Craft-repair"],
            "relationship": ["Not-in-family", "Not-in-family"],
            "race": ["White", "White"],
            "sex": ["Male", "Female"],
            "capital-gain": [0, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 40],
            "native-country": ["United-States", "United-States"],
            "income": ["<=50K", ">50K"],
        }
    )


def test_valid_data_passes(valid_df):
    assert validate_data(valid_df) is True


def test_invalid_sex_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df["sex"] = "Unknown"
    assert validate_data(bad_df) is False


def test_invalid_income_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df["income"] = "maybe"
    assert validate_data(bad_df) is False


def test_negative_capital_gain_fails(valid_df):
    bad_df = valid_df.copy()
    bad_df["capital-gain"] = -100
    assert validate_data(bad_df) is False
