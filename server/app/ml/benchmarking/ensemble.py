"""
Ensemble Experiment Module for ReconPilot Local ML.
Evaluates soft-voting blend of top calibrated candidate pipelines
against the untouched holdout evaluation split.
"""

from typing import Any

import numpy as np
import pandas as pd

from app.ml.feature_schema import TARGET_LABELS


class SoftVotingEnsemble:
    """
    Combines calibrated class probabilities from multiple diverse models.
    """

    def __init__(self, models: list[tuple[str, Any]], weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.classes_ = list(getattr(models[0][1], "classes_", TARGET_LABELS))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        weighted_probs = np.zeros((len(X), len(self.classes_)))

        for (_name, model), w in zip(self.models, self.weights, strict=False):
            p = model.predict_proba(X)
            weighted_probs += w * p

        # Normalize across rows
        row_sums = weighted_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return weighted_probs / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in top_indices])
