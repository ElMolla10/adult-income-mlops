"""Shared prediction helpers for persisted classifier artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_DECISION_THRESHOLD = 0.5


def get_decision_threshold(model: Any, default: float = DEFAULT_DECISION_THRESHOLD) -> float:
    """Return the saved probability threshold for positive-class predictions."""
    threshold = getattr(model, "decision_threshold_", default)
    return float(threshold)


def predict_positive_class(model: Any, X, threshold: float | None = None) -> np.ndarray:
    """Predict binary labels using predict_proba and a calibrated threshold."""
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(X)).astype(int)

    cutoff = get_decision_threshold(model) if threshold is None else float(threshold)
    positive_proba = np.asarray(model.predict_proba(X))[:, 1]
    return (positive_proba >= cutoff).astype(int)
