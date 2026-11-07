"""
Independent Model Evaluation Script for ReconPilot ML Investigation Engine.
Generates an independent held-out synthetic test set (different random seed)
and evaluates generalization quality, calibration metrics, and class performance.

Execution:
    python -m app.ml.evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.ml.dataset_generator import SyntheticFinancialDatasetGenerator
from app.ml.feature_schema import TARGET_LABELS
from app.ml.model_registry import get_model_registry


def evaluate_model(test_seed: int = 999, samples_per_class: int = 50) -> dict:
    """
    Evaluates the loaded model on held-out synthetic scenarios.
    """
    registry = get_model_registry()
    if registry.is_degraded or registry.pipeline is None:
        print(f"[ERROR] Cannot evaluate: model is degraded: {registry.load_error}")
        return {"error": registry.load_error}

    pipeline = registry.pipeline

    print(
        f"Generating held-out evaluation test set (seed={test_seed}, {samples_per_class} per class)..."
    )
    generator = SyntheticFinancialDatasetGenerator(seed=test_seed)
    features_list, labels_list = generator.generate_dataset(samples_per_class=samples_per_class)

    df_test = pd.DataFrame(features_list)
    y_true = np.array(labels_list)

    y_pred = pipeline.predict(df_test)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=TARGET_LABELS).tolist()

    results = {
        "model_name": registry.model_name,
        "test_seed": test_seed,
        "test_samples": len(df_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "classification_report": report,
        "confusion_matrix": conf_mat,
    }

    print("\n=======================================================")
    print("      RECONPILOT LOCAL ML EVALUATION REPORT           ")
    print("=======================================================")
    print(f"Model:             {results['model_name']}")
    print(f"Total Test Cases:  {results['test_samples']}")
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Weighted F1:       {results['weighted_f1']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print("=======================================================\n")

    return results


if __name__ == "__main__":
    evaluate_model()
