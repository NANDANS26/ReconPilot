"""
Training Script for ReconPilot Local ML Root-Cause Classifier.
Builds, evaluates, and persists a reproducible scikit-learn multiclass pipeline.
Execution:
    python -m app.ml.training
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.dataset_generator import SyntheticFinancialDatasetGenerator
from app.ml.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TARGET_LABELS,
)


def build_pipeline() -> Pipeline:
    """Builds a complete preprocessing and classification pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(
    samples_per_class: int = 150,
    artifact_dir: str = "server/app/ml/artifacts",
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Generates synthetic data, trains the pipeline, validates quality gates,
    and persists the artifact and metadata.
    """
    print(
        f"Generating synthetic dataset with {samples_per_class} samples per class across {len(TARGET_LABELS)} categories..."
    )
    generator = SyntheticFinancialDatasetGenerator(seed=42)
    features_list, labels_list = generator.generate_dataset(samples_per_class=samples_per_class)

    df_X = pd.DataFrame(features_list)
    y = np.array(labels_list)

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Compute metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS).tolist()

    print("\n--- Model Evaluation Summary ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Enforce quality gate
    if acc < 0.85 or macro_f1 < 0.85:
        raise RuntimeError(
            f"Model quality gate failed: Accuracy {acc:.4f} < 0.85 or Macro F1 {macro_f1:.4f} < 0.85"
        )

    # Ensure artifact directory exists
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "root_cause_classifier.joblib")
    metadata_path = os.path.join(artifact_dir, "root_cause_classifier_metadata.json")

    joblib.dump(pipeline, artifact_path, compress=3)
    print(f"Model artifact saved to: {artifact_path}")

    metadata = {
        "model_name": "reconpilot-rf-v1",
        "algorithm": "RandomForestClassifier",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classes": TARGET_LABELS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
        },
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_data_provenance": "Synthetic financial exception invariant generator seed 42",
        "disclaimer": "Trained on synthetic financial scenarios. Does not assert real-world production accuracy.",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

    return pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()
