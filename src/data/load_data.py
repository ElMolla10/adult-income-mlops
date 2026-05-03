"""
Stage 1: prepare
Loads adult.data and adult.test, fixes labels, saves clean CSVs.
"""
import os
import pandas as pd
import yaml


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_adult_data(params: dict):
    """Load raw adult income dataset files."""
    columns = params["data"]["columns"]

    train_df = pd.read_csv(
        params["data"]["raw_train"],
        names=columns,
        skipinitialspace=True,
        na_values="?",
    )

    # Test file has a junk header line and labels end with "."
    test_df = pd.read_csv(
        params["data"]["raw_test"],
        names=columns,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    target = params["data"]["target_column"]
    test_df[target] = test_df[target].str.rstrip(".")

    return train_df, test_df


def prepare_data() -> None:
    params = load_params()
    train_df, test_df = load_adult_data(params)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(params["data"]["processed_train"], index=False)
    test_df.to_csv(params["data"]["processed_test"], index=False)

    print(f"Train shape   : {train_df.shape}")
    print(f"Test  shape   : {test_df.shape}")
    print(f"Train missing : {train_df.isnull().sum().sum()}")
    print(f"Test  missing : {test_df.isnull().sum().sum()}")
    print("Stage 1 (prepare) complete.")


if __name__ == "__main__":
    prepare_data()
