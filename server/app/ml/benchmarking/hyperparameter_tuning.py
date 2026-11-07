"""
Bounded Hyperparameter Search for Candidate Models.
Executes stratified cross-validation on training data optimizing for Macro F1.
Avoids excessive combinatorial explosion to guarantee quick, reproducible local execution.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from app.ml.benchmarking.candidates import ReconPilotCandidateWrapper
from app.ml.benchmarking.metrics import evaluate_predictions

# Bounded parameter grids for reproducible local CPU search
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "random_forest": [
        {"n_estimators": 80, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 12, "min_samples_split": 2},
    ],
    "hist_gradient_boosting": [
        {"max_iter": 50, "max_depth": 6, "learning_rate": 0.08},
        {"max_iter": 80, "max_depth": 6, "learning_rate": 0.08},
    ],
    "xgboost": [
        {"n_estimators": 60, "max_depth": 5, "learning_rate": 0.08},
        {"n_estimators": 80, "max_depth": 5, "learning_rate": 0.08},
    ],
    "lightgbm": [
        {"n_estimators": 60, "max_depth": 5, "num_leaves": 24, "learning_rate": 0.08},
        {"n_estimators": 80, "max_depth": 5, "num_leaves": 24, "learning_rate": 0.08},
    ],
    "catboost": [
        {"iterations": 60, "depth": 5, "learning_rate": 0.08},
        {"iterations": 80, "depth": 5, "learning_rate": 0.08},
    ],
}


def tune_candidate(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: list[str],
    n_splits: int = 3,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """
    Performs Stratified K-Fold search over bounded parameter grid.
    Returns (best_params, best_macro_f1, search_history).
    """
    grid = PARAM_GRIDS.get(model_type, [{}])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_macro_f1 = -1.0
    best_params = grid[0]
    search_history = []

    print(
        f"Tuning {model_type} across {len(grid)} candidate configurations ({n_splits}-fold CV)..."
    )

    for idx, params in enumerate(grid):
        fold_f1s = []
        t0 = time.perf_counter()

        for _fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], [y_train[i] for i in train_idx]
            X_va, y_va = X_train.iloc[val_idx], [y_train[i] for i in val_idx]

            wrapper = ReconPilotCandidateWrapper(model_type=model_type, **params)
            wrapper.fit(X_tr, y_tr)

            y_va_pred = wrapper.predict(X_va)
            y_va_proba = wrapper.predict_proba(X_va)

            eval_res = evaluate_predictions(y_va, y_va_pred, y_va_proba, classes=wrapper.classes_)
            fold_f1s.append(eval_res["macro_f1"])

        mean_f1 = float(np.mean(fold_f1s))
        duration = round(time.perf_counter() - t0, 2)

        entry = {
            "config_index": idx,
            "params": params,
            "mean_macro_f1": round(mean_f1, 4),
            "fold_f1s": [round(f, 4) for f in fold_f1s],
            "cv_duration_seconds": duration,
        }
        search_history.append(entry)
        print(f"  Config {idx + 1}/{len(grid)}: Macro F1 = {mean_f1:.4f} ({duration}s)")

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_params = params

    return best_params, best_macro_f1, {"history": search_history, "best_params": best_params}
