"""
Data validation using Pandera.
Called both directly and in the CI data-validation stage.
"""
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import yaml


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


schema = DataFrameSchema(
    {
        "age": Column(int, Check.in_range(17, 100)),
        "workclass": Column(str, nullable=True),
        "fnlwgt": Column(int, Check.greater_than(0)),
        "education": Column(str),
        "education-num": Column(int, Check.in_range(1, 16)),
        "marital-status": Column(str),
        "occupation": Column(str, nullable=True),
        "relationship": Column(str),
        "race": Column(str),
        "sex": Column(str, Check.isin(["Male", "Female"])),
        "capital-gain": Column(int, Check.greater_than_or_equal_to(0)),
        "capital-loss": Column(int, Check.greater_than_or_equal_to(0)),
        "hours-per-week": Column(int, Check.in_range(1, 99)),
        "native-country": Column(str, nullable=True),
        "income": Column(str, Check.isin(["<=50K", ">50K"])),
    }
)


def validate_data(df: pd.DataFrame) -> bool:
    """Validate dataframe against schema. Returns True if valid."""
    try:
        schema.validate(df, lazy=True)
        print("Data validation passed.")
        return True
    except pa.errors.SchemaErrors as exc:
        print(f"Validation errors:\n{exc.failure_cases}")
        return False


if __name__ == "__main__":
    params = load_params()
    df = pd.read_csv(params["data"]["processed_train"])
    result = validate_data(df)
    if not result:
        raise SystemExit("Data validation failed.")
