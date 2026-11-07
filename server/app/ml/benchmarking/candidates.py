"""
Model Candidates & Preprocessing Pipeline Definitions for Multi-Algorithm Benchmarking.
Implements:
1. RandomForestClassifier (Baseline)
2. HistGradientBoostingClassifier (Scikit-learn Gradient Boosting)
3. XGBClassifier (XGBoost)
4. LGBMClassifier (LightGBM)
5. CatBoostClassifier (CatBoost)
Provides a unified wrapper interface ensuring identical input/output contracts.
"""

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.ml.feature_schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_LABELS


def create_preprocessor() -> ColumnTransformer:
    """Builds standard ColumnTransformer for numeric and categorical financial features."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )


class ReconPilotCandidateWrapper(BaseEstimator, ClassifierMixin):
    """
    Standardized wrapper ensuring all 5 algorithms:
    - Accept identical pandas DataFrame feature schema
    - Produce string class predictions conforming to TARGET_LABELS
    - Return sorted (N, 11) predict_proba matrices
    - Work natively with joblib serialization and scikit-learn inspection
    """

    def __init__(self, model_type: str = "random_forest", **kwargs: Any):
        self.model_type = model_type
        self.kwargs = kwargs
        self.preprocessor = create_preprocessor()
        self.estimator = self._init_estimator(model_type, kwargs)
        self.label_encoder = LabelEncoder()
        self.classes_ = list(TARGET_LABELS)

    def _init_estimator(self, model_type: str, kwargs: dict[str, Any]) -> Any:
        # Default fixed hyperparameters with thread and seed controls
        if model_type == "random_forest":
            params = {
                "n_estimators": 100,
                "max_depth": 12,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": 4,
            }
            params.update(kwargs)
            return RandomForestClassifier(**params)

        elif model_type == "hist_gradient_boosting":
            params = {
                "max_iter": 100,
                "max_depth": 8,
                "learning_rate": 0.1,
                "random_state": 42,
                "class_weight": "balanced",
            }
            params.update(kwargs)
            return HistGradientBoostingClassifier(**params)

        elif model_type == "xgboost":
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "n_jobs": 4,
                "eval_metric": "mlogloss",
            }
            params.update(kwargs)
            return XGBClassifier(**params)

        elif model_type == "lightgbm":
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": 4,
                "class_weight": "balanced",
                "verbose": -1,
            }
            params.update(kwargs)
            return LGBMClassifier(**params)

        elif model_type == "catboost":
            params = {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "thread_count": 4,
                "verbose": 0,
            }
            params.update(kwargs)
            return CatBoostClassifier(**params)

        else:
            raise ValueError(f"Unknown candidate model type: {model_type}")

    def fit(self, X: pd.DataFrame, y: list[str] | np.ndarray) -> "ReconPilotCandidateWrapper":
        # Fit label encoder with fixed TARGET_LABELS ordering
        self.label_encoder.fit(TARGET_LABELS)
        self.classes_ = list(self.label_encoder.classes_)

        # Transform features
        X_trans = self.preprocessor.fit_transform(X)
        y_encoded = self.label_encoder.transform(y)

        # Fit estimator
        self.estimator.fit(X_trans, y_encoded)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        y_pred_encoded = self.estimator.predict(X_trans)
        if y_pred_encoded.ndim > 1:
            y_pred_encoded = y_pred_encoded.ravel()
        return self.label_encoder.inverse_transform(y_pred_encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_trans = self.preprocessor.transform(X)
        proba = self.estimator.predict_proba(X_trans)
        return np.asarray(proba, dtype=float)


def get_all_candidate_factories() -> dict[str, Any]:
    """Returns factory mapping for all 5 candidate algorithms."""
    return {
        "random_forest": lambda **kw: ReconPilotCandidateWrapper("random_forest", **kw),
        "hist_gradient_boosting": lambda **kw: ReconPilotCandidateWrapper(
            "hist_gradient_boosting", **kw
        ),
        "xgboost": lambda **kw: ReconPilotCandidateWrapper("xgboost", **kw),
        "lightgbm": lambda **kw: ReconPilotCandidateWrapper("lightgbm", **kw),
        "catboost": lambda **kw: ReconPilotCandidateWrapper("catboost", **kw),
    }
