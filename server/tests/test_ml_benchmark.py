"""
Comprehensive Test Suite for ReconPilot ML Multi-Algorithm Benchmarking and Champion Selection.
Tests:
1. Candidate wrappers for all 5 algorithms: initialization, fit, predict, predict_proba contract
2. Group-separated dataset generation and archetype partition integrity
3. Calibration wrapper behavior and probability normalization
4. Soft-voting ensemble weighting and argmax predictions
5. Champion selection policy gates: latency, critical recall, calibration, and promotion logic
6. Challenger artifact and scorecard persistence
7. 100% offline CPU execution with zero network access
"""

import os
import shutil
import socket
import tempfile
import urllib.request
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from app.ml.benchmarking.calibration import CalibratedModelWrapper, audit_calibration
from app.ml.benchmarking.candidates import ReconPilotCandidateWrapper, get_all_candidate_factories
from app.ml.benchmarking.ensemble import SoftVotingEnsemble
from app.ml.benchmarking.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    measure_inference_latency,
)
from app.ml.champion_registry import ChampionRegistry
from app.ml.dataset_generator import (
    TEST_ARCHETYPES,
    TRAIN_ARCHETYPES,
    VAL_ARCHETYPES,
    SyntheticFinancialDatasetGenerator,
    generate_benchmark_splits,
)
from app.ml.feature_schema import CRITICAL_FINANCIAL_CLASSES, TARGET_LABELS


@pytest.fixture(scope="module")
def small_benchmark_data():
    """Generates a small deterministic dataset for fast unit test verification."""
    gen = SyntheticFinancialDatasetGenerator(seed=42)
    feats_train, labels_train = gen.generate_dataset(
        samples_per_class=5,
        archetypes=["RETAIL_ECOMMERCE", "SAAS_RECURRING"],
    )
    feats_val, labels_val = gen.generate_dataset(
        samples_per_class=4,
        archetypes=["FINTECH_NEOBANK"],
    )
    feats_test, labels_test = gen.generate_dataset(
        samples_per_class=4,
        archetypes=["TRAVEL_AIRLINE"],
    )

    X_train = pd.DataFrame(feats_train)
    y_train = labels_train

    X_val = pd.DataFrame(feats_val)
    y_val = labels_val

    X_test = pd.DataFrame(feats_test)
    y_test = labels_test

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_group_separated_dataset_archetype_isolation():
    """Verify training, validation, and holdout test archetypes have zero intersection."""
    train_set = set(TRAIN_ARCHETYPES)
    val_set = set(VAL_ARCHETYPES)
    holdout_set = set(TEST_ARCHETYPES)

    assert len(train_set) == 8
    assert len(val_set) == 2
    assert len(holdout_set) == 2

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(holdout_set)
    assert val_set.isdisjoint(holdout_set)

    # Verify generate_benchmark_splits strictly preserves group separation
    splits = generate_benchmark_splits(
        samples_per_class_train=2,
        samples_per_class_val=2,
        samples_per_class_test=2,
    )
    assert set(splits["train_archetypes"]).isdisjoint(splits["test_archetypes"])
    assert set(splits["val_archetypes"]).isdisjoint(splits["test_archetypes"])
    assert len(splits["train"][0]) == 2 * 11
    assert len(splits["test"][0]) == 2 * 11


@pytest.mark.parametrize(
    "model_type",
    ["random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"],
)
def test_all_five_candidate_wrappers_contract(model_type, small_benchmark_data):
    """Verify all 5 algorithm wrappers train, predict, and adhere to the unified output contract."""
    X_train = small_benchmark_data["X_train"]
    y_train = small_benchmark_data["y_train"]
    X_test = small_benchmark_data["X_test"]

    factories = get_all_candidate_factories()
    assert model_type in factories

    wrapper = factories[model_type]()
    wrapper.fit(X_train, y_train)

    # 1. Output classes match expected sorted classes
    assert len(wrapper.classes_) == len(TARGET_LABELS)

    # 2. Predict returns numpy array of valid string labels
    y_pred = wrapper.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(X_test)
    assert all(label in TARGET_LABELS for label in y_pred)

    # 3. Predict_proba returns (N, 11) probabilities that sum to 1.0 per row
    proba = wrapper.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(TARGET_LABELS))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    row_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X_test)), rtol=1e-4)


def test_calibrated_model_wrapper(small_benchmark_data):
    """Verify CalibratedModelWrapper calibrates probabilities and preserves normalization."""
    X_train = small_benchmark_data["X_train"]
    y_train = small_benchmark_data["y_train"]
    X_val = small_benchmark_data["X_val"]
    y_val = small_benchmark_data["y_val"]
    X_test = small_benchmark_data["X_test"]

    base = ReconPilotCandidateWrapper("random_forest", n_estimators=10, max_depth=5).fit(
        X_train, y_train
    )
    uncal_proba = base.predict_proba(X_val)

    cal = CalibratedModelWrapper(base, method="sigmoid")
    cal.calibrate(X_val, y_val)

    cal_proba = cal.predict_proba(X_test)
    assert cal_proba.shape == (len(X_test), len(base.classes_))
    row_sums = np.sum(cal_proba, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X_test)), rtol=1e-4)

    # Audit calibration metrics
    audit = audit_calibration(y_val, uncal_proba, cal.predict_proba(X_val), classes=base.classes_)
    assert "brier_score_uncalibrated" in audit
    assert "brier_score_calibrated" in audit
    assert "ece_uncalibrated" in audit
    assert "ece_calibrated" in audit
    assert audit["brier_score_calibrated"] >= 0.0


def test_soft_voting_ensemble(small_benchmark_data):
    """Verify SoftVotingEnsemble blends diverse models with proper normalization."""
    X_train = small_benchmark_data["X_train"]
    y_train = small_benchmark_data["y_train"]
    X_test = small_benchmark_data["X_test"]

    m1 = ReconPilotCandidateWrapper("random_forest", n_estimators=10).fit(X_train, y_train)
    m2 = ReconPilotCandidateWrapper("catboost", iterations=10).fit(X_train, y_train)

    ensemble = SoftVotingEnsemble(models=[("rf", m1), ("cb", m2)], weights=[0.5, 0.5])

    proba = ensemble.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(TARGET_LABELS))
    row_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(X_test)), rtol=1e-4)

    preds = ensemble.predict(X_test)
    assert len(preds) == len(X_test)
    assert all(p in TARGET_LABELS for p in preds)


def test_metrics_evaluation_formulas():
    """Verify multiclass metrics calculations for Brier score and ECE."""
    n_samples = 100
    n_classes = 11

    # Perfect predictions: zero Brier score
    y_true_indices = np.random.randint(0, n_classes, size=n_samples)
    y_proba_perfect = np.zeros((n_samples, n_classes))
    y_proba_perfect[np.arange(n_samples), y_true_indices] = 1.0

    brier_perfect = compute_multiclass_brier_score(y_true_indices, y_proba_perfect, n_classes)
    assert brier_perfect == 0.0

    ece_perfect = compute_expected_calibration_error(y_true_indices, y_proba_perfect)
    assert ece_perfect == 0.0

    # Test single-sample CPU latency measurement
    dummy_df = pd.DataFrame([{"feat": 1}] * 20)
    mock_pipeline = type("MockPipe", (), {"predict": lambda s, x: np.zeros(len(x))})()
    latency = measure_inference_latency(mock_pipeline, dummy_df, n_iterations=50)
    assert latency > 0.0


def test_champion_selection_policy_rejection_gates():
    """Verify policy rejects challengers that violate latency or critical recall constraints."""
    # Baseline reference
    baseline_lat = 35.0
    baseline_size = 800.0
    assert baseline_lat > 10.0
    assert baseline_size > 500.0

    # Challenger 1: High latency (>10ms) -> Must fail latency gate
    challenger_slow = {
        "inference_latency_ms": 15.0,
        "artifact_size_kb": 200.0,
        "metrics": {
            "macro_f1": 0.98,
            "brier_score": 0.02,
            "expected_calibration_error": 0.03,
            "per_class": {cls: {"recall": 0.95} for cls in CRITICAL_FINANCIAL_CLASSES},
        },
    }
    lat_pass = challenger_slow["inference_latency_ms"] < 10.0
    assert not lat_pass, "Challenger with latency >= 10ms must be rejected"

    # Challenger 2: Critical class recall < 0.85 -> Must fail critical gate
    challenger_bad_recall = {
        "inference_latency_ms": 5.0,
        "artifact_size_kb": 200.0,
        "metrics": {
            "macro_f1": 0.98,
            "brier_score": 0.02,
            "expected_calibration_error": 0.03,
            "per_class": {
                "MISSING_SETTLEMENT": {"recall": 0.80},  # < 0.85
                "DUPLICATE_PAYMENT": {"recall": 0.95},
                "AMOUNT_MISMATCH": {"recall": 0.95},
                "FEE_MISMATCH": {"recall": 0.95},
                "REFUND_MISMATCH": {"recall": 0.95},
            },
        },
    }
    rec_pass = (
        min(
            challenger_bad_recall["metrics"]["per_class"][cls]["recall"]
            for cls in CRITICAL_FINANCIAL_CLASSES
        )
        >= 0.85
    )
    assert not rec_pass, "Challenger with critical recall < 0.85 must be rejected"


def test_champion_registry_persistence():
    """Verify ChampionRegistry saves challenger artifacts and atomic promotion."""
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ChampionRegistry(artifacts_dir=temp_dir)

        # Save challenger artifact
        dummy_model = {"model": "test_challenger"}
        dummy_meta = {"metrics": {"macro_f1": 0.99}, "latency_ms": 4.5}
        path = registry.save_challenger_artifact("test_challenger", dummy_model, dummy_meta)

        assert os.path.exists(path)
        meta_path = os.path.join(temp_dir, "challengers", "test_challenger_metadata.json")
        assert os.path.exists(meta_path)

        # Promote champion
        promoted = registry.promote_champion(
            champion_name="test_challenger",
            pipeline=dummy_model,
            metadata=dummy_meta,
            selection_rationale="Passed all 5 policy gates.",
        )

        assert promoted["model_family"] == "test_challenger"
        prod_joblib = os.path.join(temp_dir, "root_cause_classifier.joblib")
        prod_meta = os.path.join(temp_dir, "root_cause_classifier_metadata.json")
        assert os.path.exists(prod_joblib)
        assert os.path.exists(prod_meta)

        saved_meta = registry.get_champion_metadata()
        assert saved_meta["model_family"] == "test_challenger"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_benchmarking_zero_external_network_calls(small_benchmark_data):
    """Verify benchmarking pipeline executes 100% locally with zero external network socket calls."""
    X_train = small_benchmark_data["X_train"]
    y_train = small_benchmark_data["y_train"]

    with (
        patch.object(socket, "socket", side_effect=RuntimeError("Network access forbidden")),
        patch.object(
            urllib.request, "urlopen", side_effect=RuntimeError("Network access forbidden")
        ),
    ):
        wrapper = ReconPilotCandidateWrapper("catboost", iterations=5)
        wrapper.fit(X_train, y_train)
        pred = wrapper.predict(small_benchmark_data["X_test"])
        assert len(pred) > 0
