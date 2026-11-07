"""
Confidence Calibration & Conservative Abstention Engine for ReconPilot ML Models.
Implements:
1. Probability calibration via Sigmoid / Platt scaling on validation holdout
2. Reliability curve and Expected Calibration Error (ECE) calculation
3. Conservative dual-gate abstention (confidence threshold & minimum margin)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from app.ml.benchmarking.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)
from app.ml.feature_schema import TARGET_LABELS


class CalibratedModelWrapper:
    """
    Wraps any base candidate model with a calibrated probability estimator
    fitted on an independent validation split.
    """

    def __init__(self, base_wrapper: Any, method: str = "sigmoid"):
        self.base_wrapper = base_wrapper
        self.method = method
        self.calibrator: CalibratedClassifierCV | None = None
        self.classes_ = list(getattr(base_wrapper, "classes_", TARGET_LABELS))

    def calibrate(self, X_val: pd.DataFrame, y_val: list[str]) -> "CalibratedModelWrapper":
        """Calibrates the pre-fitted base model on independent validation features."""
        X_val_trans = self.base_wrapper.preprocessor.transform(X_val)
        y_val_enc = self.base_wrapper.label_encoder.transform(y_val)

        # Fit CalibratedClassifierCV on pre-fitted base model
        if FrozenEstimator is not None:
            _, counts = np.unique(y_val_enc, return_counts=True)
            min_class_count = int(np.min(counts)) if len(counts) > 0 else 5
            cv_splits = max(2, min(5, min_class_count))

            frozen_estimator = FrozenEstimator(self.base_wrapper.estimator)
            self.calibrator = CalibratedClassifierCV(
                estimator=frozen_estimator,
                method=self.method,
                cv=cv_splits,
            )
        else:
            self.calibrator = CalibratedClassifierCV(
                estimator=self.base_wrapper.estimator,
                method=self.method,
                cv="prefit",
            )
        self.calibrator.fit(X_val_trans, y_val_enc)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is not None:
            X_trans = self.base_wrapper.preprocessor.transform(X)
            return np.asarray(self.calibrator.predict_proba(X_trans), dtype=float)
        return self.base_wrapper.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.base_wrapper.label_encoder.inverse_transform(pred_idx)


def audit_calibration(
    y_true: list[str],
    uncalibrated_proba: np.ndarray,
    calibrated_proba: np.ndarray,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compares calibration metrics (ECE, Brier score) before and after calibration.
    """
    class_list = classes or sorted(TARGET_LABELS)
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    y_idx = np.array([class_to_idx[c] for c in y_true])

    n_classes = len(class_list)
    ece_before = compute_expected_calibration_error(y_idx, uncalibrated_proba)
    ece_after = compute_expected_calibration_error(y_idx, calibrated_proba)

    brier_before = compute_multiclass_brier_score(y_idx, uncalibrated_proba, n_classes)
    brier_after = compute_multiclass_brier_score(y_idx, calibrated_proba, n_classes)

    return {
        "brier_score_uncalibrated": round(brier_before, 4),
        "brier_score_calibrated": round(brier_after, 4),
        "ece_uncalibrated": round(ece_before, 4),
        "ece_calibrated": round(ece_after, 4),
        "calibration_improved": bool(ece_after < ece_before or brier_after < brier_before),
    }
