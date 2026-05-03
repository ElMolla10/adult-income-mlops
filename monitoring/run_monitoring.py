"""
Monitoring script.
Generates two Evidently HTML reports:
  1. baseline_report.html  — reference vs clean held-out set (minimal drift)
  2. drift_report.html     — reference vs subgroup-filtered production set (real drift)

If drift ratio > threshold, writes drift_alert.json and logs a structured warning.
"""
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.getcwd())


def load_params() -> dict:
    with open("configs/params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> Report:
    report = Report(
        metrics=[DataDriftPreset(), DataQualityPreset(), ClassificationPreset()]
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    return report


def save_report(report: Report, name: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.html")
    report.save_html(path)
    return path


def extract_drift_info(report: Report) -> tuple[float, list[str]]:
    """Returns (drift_ratio, list_of_drifted_feature_names)."""
    result = report.as_dict()
    drifted = []
    total = 0

    for metric in result.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            drift_by_col = metric.get("result", {}).get("drift_by_columns", {})
            for feature, data in drift_by_col.items():
                total += 1
                if data.get("drift_detected", False):
                    drifted.append(feature)

    ratio = len(drifted) / total if total > 0 else 0.0
    return ratio, drifted


def run_monitoring() -> None:
    params = load_params()
    target = params["data"]["target_column"]
    output_dir = params["monitoring"]["reports_dir"]
    threshold = params["monitoring"]["drift_threshold"]
    positive_label = params["data"]["positive_label"]

    reference_df = pd.read_csv(params["monitoring"]["reference_data"])
    production_df = pd.read_csv(params["monitoring"]["production_data"])

    with open(params["preprocessing"]["pipeline_path"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(params["preprocessing"]["best_model_path"], "rb") as f:
        model = pickle.load(f)

    feature_cols = [c for c in reference_df.columns if c != target]

    reference_df = reference_df.copy()
    production_df = production_df.copy()
    negative_label = "<=50K"
    reference_pred = model.predict(preprocessor.transform(reference_df[feature_cols]))
    production_pred = model.predict(preprocessor.transform(production_df[feature_cols]))
    reference_df["prediction"] = [
        positive_label if pred == 1 else negative_label for pred in reference_pred
    ]
    production_df["prediction"] = [
        positive_label if pred == 1 else negative_label for pred in production_pred
    ]

    column_mapping = ColumnMapping(
        target=target,
        prediction="prediction",
        pos_label=positive_label,
    )

    # ------------------------------------------------------------------ #
    # Report 1: Baseline (reference vs clean 20% held-out from reference)
    # Expected: minimal drift
    # ------------------------------------------------------------------ #
    split = int(len(reference_df) * 0.8)
    baseline_ref = reference_df.iloc[:split]
    baseline_cur = reference_df.iloc[split:]

    logger.info("Generating baseline report (reference vs held-out)...")
    baseline_report = build_report(baseline_ref, baseline_cur, column_mapping)
    baseline_path = save_report(baseline_report, "baseline_report", output_dir)
    logger.info(f"Baseline report saved: {baseline_path}")

    base_ratio, base_drifted = extract_drift_info(baseline_report)
    logger.info(
        f"Baseline drift ratio: {base_ratio:.2%} on {len(base_drifted)} features"
    )

    # ------------------------------------------------------------------ #
    # Report 2: Drift (reference vs subgroup-filtered production set)
    # Expected: clear drift on sex, race, and related features
    # ------------------------------------------------------------------ #
    logger.info("Generating drift report (reference vs production subset)...")
    drift_report = build_report(reference_df, production_df, column_mapping)
    drift_path = save_report(drift_report, "drift_report", output_dir)
    logger.info(f"Drift report saved: {drift_path}")

    drift_ratio, drifted_features = extract_drift_info(drift_report)
    logger.info(
        f"Production drift ratio: {drift_ratio:.2%} | "
        f"Threshold: {threshold:.2%} | "
        f"Drifted features: {drifted_features}"
    )

    # ------------------------------------------------------------------ #
    # Threshold check
    # ------------------------------------------------------------------ #
    if drift_ratio > threshold:
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_ratio": round(drift_ratio, 4),
            "threshold": threshold,
            "drifted_features": drifted_features,
            "drifted_feature_count": len(drifted_features),
            "action": "RETRAIN_REQUIRED",
        }
        logger.warning(
            f"DRIFT ALERT: {len(drifted_features)} features drifted "
            f"({drift_ratio:.2%} > {threshold:.2%} threshold). "
            f"Features: {drifted_features}"
        )
        alert_path = os.path.join(output_dir, "drift_alert.json")
        with open(alert_path, "w") as fh:
            json.dump(alert, fh, indent=2)
        logger.warning(f"Alert written to {alert_path}. Action: RETRAIN_REQUIRED")
    else:
        logger.info("No significant drift detected. System healthy.")


if __name__ == "__main__":
    run_monitoring()
