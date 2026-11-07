"""
Comprehensive Evaluation Metrics for Multi-Algorithm Model Selection.
Computes accuracy, macro/weighted/balanced metrics, multiclass log loss,
multiclass Brier score, expected calibration error (ECE), and latency.
"""

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from app.ml.feature_schema import TARGET_LABELS


def compute_multiclass_brier_score(
    y_true_indices: np.ndarray, y_proba: np.ndarray, n_classes: int
) -> float:
    """
    Computes multiclass Brier score:
    Brier = (1 / N) * sum_{i=1}^N sum_{k=1}^K (y_{ik} - p_{ik})^2
    """
    N = len(y_true_indices)
    one_hot = np.zeros((N, n_classes))
    one_hot[np.arange(N), y_true_indices] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_expected_calibration_error(
    y_true_indices: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multiclass classification.
    Bins predictions by top confidence and measures weighted gap between accuracy and confidence.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = predictions == y_true_indices

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(round(ece, 4))


def measure_inference_latency(pipeline: Any, X_sample: Any, n_iterations: int = 500) -> float:
    """
    Measures average single-sample CPU inference latency in milliseconds.
    """
    # Warmup
    _ = pipeline.predict(X_sample.iloc[:10])

    latencies = []
    # Test individual single-sample predictions
    single_samples = [X_sample.iloc[[i]] for i in range(min(50, len(X_sample)))]

    for _ in range(n_iterations // len(single_samples)):
        for sample in single_samples:
            t0 = time.perf_counter()
            _ = pipeline.predict(sample)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return float(round(np.mean(latencies), 3))


def evaluate_predictions(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    y_proba: np.ndarray,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Computes complete performance, calibration, and risk metrics for a model.
    """
    class_list = classes or TARGET_LABELS
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    y_true_idx = np.array([class_to_idx[c] for c in y_true_arr])

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    rep = classification_report(
        y_true_arr, y_pred_arr, labels=class_list, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=class_list).tolist()

    # Normalize probabilities to avoid float precision sum warnings
    row_sums = np.sum(y_proba, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_proba = y_proba / row_sums

    # Probability metrics
    loss = float(log_loss(y_true_idx, norm_proba, labels=list(range(len(class_list)))))
    brier = compute_multiclass_brier_score(y_true_idx, norm_proba, len(class_list))
    ece = compute_expected_calibration_error(y_true_idx, norm_proba)

    # Per-class summary
    per_class = {}
    for cls_name in class_list:
        cls_data = rep.get(cls_name, {})
        per_class[cls_name] = {
            "precision": round(float(cls_data.get("precision", 0.0)), 4),
            "recall": round(float(cls_data.get("recall", 0.0)), 4),
            "f1_score": round(float(cls_data.get("f1-score", 0.0)), 4),
            "support": int(cls_data.get("support", 0)),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat,
    }
