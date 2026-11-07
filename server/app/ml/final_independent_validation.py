"""
ReconPilot Final Independent ML Release Validation & Stress Audit Engine.
Executes:
1. Frozen Champion Integrity Verification (Artifact SHA256 & Metadata).
2. Independent Evaluation on Fresh Scenario Distributions (Seed 20261107).
3. Complete Reliability & Calibration Audit (Accuracy, F1, Brier, ECE, Confusion Matrix).
4. Distribution-Shift & Adversarial Financial Stress Testing.
5. Benchmarking: Latency Profiling (p50, p95, p99), Throughput, Cold-Start Load Time, Degradation Fallback.
"""

import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from app.ml.benchmarking.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)
from app.ml.feature_schema import (
    CRITICAL_FINANCIAL_CLASSES,
    FEATURE_SCHEMA_VERSION,
    TARGET_LABELS,
)
from app.ml.inference import LocalMLInferenceService
from app.ml.model_registry import ModelRegistry, get_model_registry

ARTIFACT_PATH = "server/app/ml/artifacts/root_cause_classifier.joblib"
METADATA_PATH = "server/app/ml/artifacts/root_cause_classifier_metadata.json"


def get_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()


def generate_independent_test_dataset(
    seed: int = 20261107, samples_per_class: int = 100
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generates a brand-new, independent evaluation dataset using fresh random seeds
    and realistic commercial financial conditions.
    This dataset has NEVER been seen during training, tuning, or champion selection.
    """
    from app.ml.dataset_generator import SyntheticFinancialDatasetGenerator

    gen = SyntheticFinancialDatasetGenerator(seed=seed, realistic=True)
    raw_feats, labels = gen.generate_dataset(samples_per_class=samples_per_class)
    return pd.DataFrame(raw_feats), labels


def run_final_ml_release_validation(seed: int = 20261107) -> dict[str, Any]:
    print("==========================================================================")
    print("      RECONPILOT FINAL INDEPENDENT ML RELEASE VALIDATION & AUDIT         ")
    print("==========================================================================")

    # 1. Freeze & Artifact Integrity Check
    assert os.path.exists(ARTIFACT_PATH), f"Champion artifact missing at {ARTIFACT_PATH}"
    assert os.path.exists(METADATA_PATH), f"Metadata missing at {METADATA_PATH}"

    artifact_sha256 = get_sha256(ARTIFACT_PATH)
    artifact_size_kb = round(os.path.getsize(ARTIFACT_PATH) / 1024, 2)

    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Frozen Champion Artifact: {ARTIFACT_PATH}")
    print(f"Artifact Size:            {artifact_size_kb} KB")
    print(f"Artifact SHA256:          {artifact_sha256}")
    print(f"Model Name in Metadata:   {meta.get('model_name')}")
    print(f"Feature Schema Version:   {meta.get('feature_schema_version')}")
    print(f"Algorithm Family:         {meta.get('algorithm')}")

    assert meta.get("feature_schema_version") == FEATURE_SCHEMA_VERSION, "Schema version mismatch!"
    assert meta.get("model_family") == "catboost", "Champion must be CatBoost!"

    # 2. Load Active Model into Registry
    registry = get_model_registry()
    registry.reload()
    assert registry.is_loaded is True
    assert registry.is_degraded is False
    pipeline = registry.pipeline
    assert pipeline is not None

    # 3. Generate New Independent Test Dataset (Seed: 20261107)
    print("\nGenerating Brand-New Independent Evaluation Dataset (Seed: 20261107, 100/class)...")
    X_test_df, y_test = generate_independent_test_dataset(seed=20261107, samples_per_class=100)
    print(f"Dataset generated: {len(X_test_df)} samples across {len(TARGET_LABELS)} classes.")

    # 4. Predict using Frozen Champion
    print("\nRunning inference on independent test dataset...")
    y_pred = pipeline.predict(X_test_df)
    y_proba = pipeline.predict_proba(X_test_df)

    # Normalize proba for float precision
    row_sums = np.sum(y_proba, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_proba = y_proba / row_sums

    class_list = pipeline.classes_
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    y_true_idx = np.array([class_to_idx[c] for c in y_test])

    # 5. Compute Exhaustive Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    loss = float(log_loss(y_true_idx, norm_proba, labels=list(range(len(class_list)))))
    brier = compute_multiclass_brier_score(y_true_idx, norm_proba, len(class_list))
    ece = compute_expected_calibration_error(y_true_idx, norm_proba)

    rep = classification_report(
        y_test, y_pred, labels=class_list, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_test, y_pred, labels=class_list).tolist()

    print("\n==========================================================================")
    print("                     INDEPENDENT EVALUATION RESULTS                       ")
    print("==========================================================================")
    print(f"Accuracy:                 {acc:.4f}")
    print(f"Macro F1 Score:           {macro_f1:.4f}")
    print(f"Weighted F1 Score:        {weighted_f1:.4f}")
    print(f"Balanced Accuracy:        {bal_acc:.4f}")
    print(f"Multiclass Log Loss:      {loss:.4f}")
    print(f"Multiclass Brier Score:   {brier:.4f}")
    print(f"Expected Calib Error:     {ece:.4f}")

    # Per-class summary
    per_class_metrics = {}
    print("\nPer-Class Breakdown:")
    for cls in class_list:
        data = rep.get(cls, {})
        p = round(float(data.get("precision", 0.0)), 4)
        r = round(float(data.get("recall", 0.0)), 4)
        f = round(float(data.get("f1-score", 0.0)), 4)
        s = int(data.get("support", 0))
        per_class_metrics[cls] = {"precision": p, "recall": r, "f1_score": f, "support": s}
        is_crit = " [CRITICAL]" if cls in CRITICAL_FINANCIAL_CLASSES else ""
        print(f"  {cls:<25} P: {p:.4f} | R: {r:.4f} | F1: {f:.4f} | N: {s}{is_crit}")

    # 6. Critical Financial Recall Gate Validation
    crit_recalls = {cls: per_class_metrics[cls]["recall"] for cls in CRITICAL_FINANCIAL_CLASSES}
    min_crit_rec = min(crit_recalls.values())
    print(f"\nMinimum Critical Financial Recall: {min_crit_rec:.4f} (Safety Floor >= 0.8500)")
    assert min_crit_rec >= 0.85, f"CRITICAL RECALL FAILURE: {crit_recalls}"

    # 7. Distribution Shift & Adversarial Stress Testing
    print("\n==========================================================================")
    print("             DISTRIBUTION SHIFT & ADVERSARIAL STRESS TESTING              ")
    print("==========================================================================")
    service = LocalMLInferenceService(registry=registry)
    stress_results = []

    # Case A: Massive Amount Outlier (100,000,000 minor units = 1,000,000 INR)
    sample_large = X_test_df.iloc[0].to_dict()
    sample_large["payment_amount_minor"] = 100000000
    sample_large["actual_amount_minor"] = 100000000
    sample_large["settlement_gross_minor"] = 100000000
    res_large = service.predict(sample_large)
    print(
        f"1. Extreme Amount Outlier (1M INR): Result={res_large.predicted_cause}, Conf={res_large.confidence_score:.4f}"
    )
    stress_results.append(
        {
            "case": "Extreme Amount Outlier",
            "prediction": res_large.predicted_cause,
            "conf": res_large.confidence_score,
        }
    )

    # Case B: Unseen Currency (e.g. 'EUR') & Novel Merchant
    sample_novel = X_test_df.iloc[1].to_dict()
    sample_novel["currency"] = "EUR"
    sample_novel["payment_status"] = "SETTLING_EXTERNAL"
    res_novel = service.predict(sample_novel)
    print(
        f"2. Unseen Category (EUR / Novel Status): Result={res_novel.predicted_cause}, Conf={res_novel.confidence_score:.4f}"
    )
    stress_results.append(
        {
            "case": "Unseen Categorical",
            "prediction": res_novel.predicted_cause,
            "conf": res_novel.confidence_score,
        }
    )

    # Case C: Conflicting High-Variance Double-Discrepancy (Fee + Refund simultaneously)
    sample_conflict = X_test_df.iloc[2].to_dict()
    sample_conflict["fee_difference_minor"] = 2500
    sample_conflict["refund_difference_minor"] = 2500
    sample_conflict["is_ambiguous"] = 1.0
    sample_conflict["conflict_count"] = 3.0
    res_conflict = service.predict(sample_conflict)
    print(
        f"3. Conflicting Discrepancies (Fee + Refund): Result={res_conflict.predicted_cause}, Abstaining={res_conflict.is_abstaining}"
    )
    stress_results.append(
        {
            "case": "Conflicting Discrepancies",
            "prediction": res_conflict.predicted_cause,
            "abstaining": res_conflict.is_abstaining,
        }
    )

    # Case D: Zero Evidence Dossier
    sample_zero_ev = X_test_df.iloc[3].to_dict()
    sample_zero_ev["supporting_evidence_count"] = 0.0
    sample_zero_ev["has_incomplete_evidence"] = 1.0
    sample_zero_ev["missing_info_count"] = 5.0
    res_zero = service.predict(sample_zero_ev)
    print(
        f"4. Zero Evidence Dossier: Result={res_zero.predicted_cause}, Abstaining={res_zero.is_abstaining}"
    )
    stress_results.append(
        {
            "case": "Zero Evidence Dossier",
            "prediction": res_zero.predicted_cause,
            "abstaining": res_zero.is_abstaining,
        }
    )

    # 8. Fair Latency & Throughput Benchmark
    print("\n==========================================================================")
    print("             FAIR LATENCY, THROUGHPUT & COLD LOAD BENCHMARK               ")
    print("==========================================================================")
    # Cold artifact load time
    t_load_start = time.perf_counter()
    _ = ModelRegistry(artifact_path=ARTIFACT_PATH, metadata_path=METADATA_PATH)
    t_load_ms = (time.perf_counter() - t_load_start) * 1000.0

    # Warmup
    for _ in range(50):
        _ = pipeline.predict(X_test_df.iloc[[0]])

    # 1,000 Repeated Single-Sample Latency Measurements
    single_latencies = []
    test_rows = [X_test_df.iloc[[i % len(X_test_df)]] for i in range(1000)]
    for r in test_rows:
        t0 = time.perf_counter()
        _ = pipeline.predict(r)
        t1 = time.perf_counter()
        single_latencies.append((t1 - t0) * 1000.0)

    p50_lat = float(np.percentile(single_latencies, 50))
    p95_lat = float(np.percentile(single_latencies, 95))
    p99_lat = float(np.percentile(single_latencies, 99))
    mean_lat = float(np.mean(single_latencies))

    # Batch throughput (1000 samples)
    t_batch_start = time.perf_counter()
    _ = pipeline.predict(X_test_df)
    t_batch_s = time.perf_counter() - t_batch_start
    throughput_sps = round(len(X_test_df) / t_batch_s, 1)

    print(f"Cold Artifact Load Time:      {t_load_ms:.2f} ms")
    print(f"Single-Sample CPU Latency p50: {p50_lat:.3f} ms")
    print(f"Single-Sample CPU Latency p95: {p95_lat:.3f} ms")
    print(f"Single-Sample CPU Latency p99: {p99_lat:.3f} ms")
    print(f"Mean Latency:                 {mean_lat:.3f} ms")
    print(f"Batch Inference Throughput:   {throughput_sps} samples/sec")

    # 9. Degraded Mode & Fallback Safety Verification
    print("\n==========================================================================")
    print("                 SAFETY & DEGRADED FALLBACK VERIFICATION                  ")
    print("==========================================================================")
    # Corrupt path registry
    degraded_reg = ModelRegistry(artifact_path="server/app/ml/artifacts/non_existent.joblib")
    assert degraded_reg.is_degraded is True
    degraded_service = LocalMLInferenceService(registry=degraded_reg)
    deg_res = degraded_service.predict(sample_large)
    assert deg_res.is_abstaining is True
    assert deg_res.inference_mode == "DETERMINISTIC_FALLBACK"
    assert deg_res.predicted_cause == "INSUFFICIENT_EVIDENCE"
    print("Degraded mode successfully suppresses predictions and routes to DETERMINISTIC_FALLBACK.")

    results_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "evaluation_seed": seed,
        "sample_count": len(X_test_df),
        "artifact_sha256": artifact_sha256,
        "artifact_size_kb": artifact_size_kb,
        "model_name": meta.get("model_name"),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "log_loss": round(loss, 4),
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
            "min_critical_recall": round(min_crit_rec, 4),
            "per_class": per_class_metrics,
            "confusion_matrix": conf_mat,
        },
        "latency_profile_ms": {
            "cold_load_ms": round(t_load_ms, 2),
            "p50_ms": round(p50_lat, 3),
            "p95_ms": round(p95_lat, 3),
            "p99_ms": round(p99_lat, 3),
            "mean_ms": round(mean_lat, 3),
            "throughput_samples_per_sec": throughput_sps,
        },
        "stress_test_cases": stress_results,
    }

    report_path = "server/app/ml/artifacts/final_independent_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nFinal independent validation report persisted to: {report_path}")
    print("==========================================================================")
    print("                     ALL VALIDATION GATES PASSED!                         ")
    print("==========================================================================")
    return results_payload


if __name__ == "__main__":
    run_final_ml_release_validation()
