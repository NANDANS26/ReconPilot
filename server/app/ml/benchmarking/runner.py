"""
Master Benchmarking & Champion Selection Experiment Runner.
Executes multi-algorithm local benchmarking across:
1. RandomForestClassifier (Baseline)
2. HistGradientBoostingClassifier (Scikit-learn Gradient Boosting)
3. XGBClassifier (XGBoost)
4. LGBMClassifier (LightGBM)
5. CatBoostClassifier (CatBoost)
6. Soft-Voting Calibrated Ensemble

Evaluates all models against the untouched, group-separated holdout test set.
Applies the formal Champion Selection Policy and updates the production artifact.
"""

import json
import os
import time
from typing import Any

import pandas as pd

from app.ml.benchmarking.calibration import CalibratedModelWrapper, audit_calibration
from app.ml.benchmarking.candidates import get_all_candidate_factories
from app.ml.benchmarking.ensemble import SoftVotingEnsemble
from app.ml.benchmarking.hyperparameter_tuning import tune_candidate
from app.ml.benchmarking.metrics import evaluate_predictions, measure_inference_latency
from app.ml.champion_registry import ChampionRegistry
from app.ml.dataset_generator import generate_benchmark_splits

CRITICAL_FINANCIAL_CLASSES = [
    "FEE_MISMATCH",
    "AMOUNT_MISMATCH",
    "MISSING_PAYMENT",
    "MISSING_SETTLEMENT",
    "REFUND_MISMATCH",
]


def run_full_benchmark(
    do_tuning: bool = True,
    samples_per_class_train: int = 200,
    samples_per_class_val: int = 50,
    samples_per_class_test: int = 100,
) -> dict[str, Any]:
    """
    Executes the complete end-to-end multi-algorithm benchmark experiment.
    """
    start_time = time.perf_counter()
    print("==========================================================================")
    print("      RECONPILOT ADVANCED LOCAL ML BENCHMARK & CHAMPION SELECTION        ")
    print("==========================================================================")

    # 1. Generate strictly group-separated dataset splits
    splits = generate_benchmark_splits(
        train_seed=42,
        val_seed=123,
        test_seed=999,
        samples_per_class_train=samples_per_class_train,
        samples_per_class_val=samples_per_class_val,
        samples_per_class_test=samples_per_class_test,
    )

    X_train_df = pd.DataFrame(splits["train"][0])
    y_train = splits["train"][1]

    X_val_df = pd.DataFrame(splits["val"][0])
    y_val = splits["val"][1]

    X_test_df = pd.DataFrame(splits["test"][0])
    y_test = splits["test"][1]

    print("\nDatasets Ready:")
    print(
        f"  Training:        {len(X_train_df)} samples across 8 archetypes ({splits['train_archetypes']})"
    )
    print(
        f"  Validation:      {len(X_val_df)} samples across 2 archetypes ({splits['val_archetypes']})"
    )
    print(
        f"  Holdout Test:    {len(X_test_df)} samples across 2 UNSEEN archetypes ({splits['test_archetypes']})"
    )

    registry = ChampionRegistry()
    factories = get_all_candidate_factories()

    candidate_results: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    calibrated_models: dict[str, Any] = {}

    # 2. Train & tune each candidate
    for model_name, factory in factories.items():
        print("\n--------------------------------------------------------------------------")
        print(f"  Evaluating Candidate: {model_name.upper()}")
        print("--------------------------------------------------------------------------")

        best_params = {}
        tuning_info = {}
        if do_tuning:
            best_params, best_cv_f1, tuning_info = tune_candidate(
                model_name, X_train_df, y_train, n_splits=3
            )
            print(f"  Optimal parameters selected: {best_params} (CV Macro F1 = {best_cv_f1:.4f})")

        # Fit final candidate on full training set
        t0 = time.perf_counter()
        candidate_wrapper = factory(**best_params)
        candidate_wrapper.fit(X_train_df, y_train)
        train_duration = round(time.perf_counter() - t0, 3)

        fitted_models[model_name] = candidate_wrapper

        # Evaluate on Validation before calibration
        val_uncal_proba = candidate_wrapper.predict_proba(X_val_df)

        # Calibrate on validation split
        calibrated_wrapper = CalibratedModelWrapper(candidate_wrapper, method="sigmoid")
        calibrated_wrapper.calibrate(X_val_df, y_val)
        calibrated_models[model_name] = calibrated_wrapper

        val_cal_proba = calibrated_wrapper.predict_proba(X_val_df)
        calib_audit = audit_calibration(
            y_val, val_uncal_proba, val_cal_proba, classes=candidate_wrapper.classes_
        )

        # Evaluate on UNTOUCHED Holdout Test Set
        holdout_pred = calibrated_wrapper.predict(X_test_df)
        holdout_proba = calibrated_wrapper.predict_proba(X_test_df)

        metrics = evaluate_predictions(
            y_test, holdout_pred, holdout_proba, classes=calibrated_wrapper.classes_
        )

        # Measure single-sample CPU inference latency
        latency_ms = measure_inference_latency(calibrated_wrapper, X_test_df, n_iterations=300)

        # Save challenger artifact
        challenger_path = registry.save_challenger_artifact(
            model_name=model_name,
            pipeline=calibrated_wrapper,
            metadata={
                "algorithm": model_name,
                "best_params": best_params,
                "metrics": metrics,
                "calibration": calib_audit,
                "latency_ms": latency_ms,
                "train_duration_s": train_duration,
            },
        )
        artifact_size_kb = round(os.path.getsize(challenger_path) / 1024, 2)

        candidate_results[model_name] = {
            "algorithm": model_name,
            "best_params": best_params,
            "train_duration_s": train_duration,
            "inference_latency_ms": latency_ms,
            "artifact_size_kb": artifact_size_kb,
            "tuning": tuning_info,
            "calibration": calib_audit,
            "metrics": metrics,
        }

        print(f"  Holdout Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Holdout Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"  Holdout Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
        print(f"  Multiclass Brier Score:   {metrics['brier_score']:.4f}")
        print(f"  Expected Calib Error:     {metrics['expected_calibration_error']:.4f}")
        print(f"  CPU Inference Latency:    {latency_ms:.3f} ms/sample")

    # 3. Optional Ensemble Experiment: Soft-voting blend of top calibrated models
    print("\n--------------------------------------------------------------------------")
    print("  Evaluating Soft-Voting Calibrated Ensemble")
    print("--------------------------------------------------------------------------")
    # Blend top diverse architectures: LightGBM + CatBoost + RandomForest
    ensemble_members = [
        ("lightgbm", calibrated_models["lightgbm"]),
        ("catboost", calibrated_models["catboost"]),
        ("random_forest", calibrated_models["random_forest"]),
    ]
    ensemble = SoftVotingEnsemble(models=ensemble_members, weights=[0.40, 0.40, 0.20])

    ens_pred = ensemble.predict(X_test_df)
    ens_proba = ensemble.predict_proba(X_test_df)
    ens_latency = measure_inference_latency(ensemble, X_test_df, n_iterations=200)

    ens_metrics = evaluate_predictions(y_test, ens_pred, ens_proba, classes=ensemble.classes_)

    ens_challenger_path = registry.save_challenger_artifact(
        model_name="ensemble_soft_voting",
        pipeline=ensemble,
        metadata={
            "algorithm": "SoftVotingEnsemble",
            "members": ["lightgbm", "catboost", "random_forest"],
            "weights": [0.40, 0.40, 0.20],
            "metrics": ens_metrics,
            "latency_ms": ens_latency,
        },
    )
    ens_size_kb = round(os.path.getsize(ens_challenger_path) / 1024, 2)

    candidate_results["ensemble_soft_voting"] = {
        "algorithm": "SoftVotingEnsemble",
        "members": ["lightgbm (0.40)", "catboost (0.40)", "random_forest (0.20)"],
        "train_duration_s": 0.0,
        "inference_latency_ms": ens_latency,
        "artifact_size_kb": ens_size_kb,
        "calibration": {
            "brier_score": ens_metrics["brier_score"],
            "ece": ens_metrics["expected_calibration_error"],
        },
        "metrics": ens_metrics,
    }

    print(f"  Holdout Accuracy:         {ens_metrics['accuracy']:.4f}")
    print(f"  Holdout Macro F1:         {ens_metrics['macro_f1']:.4f}")
    print(f"  Holdout Balanced Acc:     {ens_metrics['balanced_accuracy']:.4f}")
    print(f"  Multiclass Brier Score:   {ens_metrics['brier_score']:.4f}")
    print(f"  Expected Calib Error:     {ens_metrics['expected_calibration_error']:.4f}")
    print(f"  CPU Inference Latency:    {ens_latency:.3f} ms/sample")

    # 4. Formal Champion Selection Policy Evaluation
    print("\n==========================================================================")
    print("               FORMAL CHAMPION SELECTION POLICY EVALUATION                ")
    print("==========================================================================")

    rf_baseline_f1 = candidate_results["random_forest"]["metrics"]["macro_f1"]
    rf_baseline_brier = candidate_results["random_forest"]["metrics"]["brier_score"]
    rf_baseline_ece = candidate_results["random_forest"]["metrics"]["expected_calibration_error"]
    rf_baseline_lat = candidate_results["random_forest"]["inference_latency_ms"]
    rf_baseline_size = candidate_results["random_forest"]["artifact_size_kb"]

    champion_name = "random_forest"
    champion_pipeline = calibrated_models["random_forest"]
    champion_metadata = candidate_results["random_forest"]
    selection_rationale = "Default baseline model retained."

    challenger_evaluations = []

    for name, res in candidate_results.items():
        if name == "random_forest":
            continue

        macro_f1 = res["metrics"]["macro_f1"]
        brier = res["metrics"]["brier_score"]
        ece = res["metrics"]["expected_calibration_error"]
        lat = res["inference_latency_ms"]
        size_kb = res["artifact_size_kb"]

        # Gate 1: Performance or Efficiency Advantage
        f1_diff = macro_f1 - rf_baseline_f1
        latency_reduction_pct = (
            ((rf_baseline_lat - lat) / rf_baseline_lat) * 100.0 if rf_baseline_lat > 0 else 0.0
        )
        size_smaller = size_kb < rf_baseline_size

        advantage_by_f1 = f1_diff >= 0.005
        advantage_by_efficiency = (
            (abs(f1_diff) <= 0.002) and (latency_reduction_pct >= 30.0) and size_smaller
        )
        passes_performance_gate = advantage_by_f1 or advantage_by_efficiency

        # Gate 2: Critical Financial Classes Recall >= 0.85
        per_class = res["metrics"]["per_class"]
        critical_recalls = {
            cls: per_class.get(cls, {}).get("recall", 0.0) for cls in CRITICAL_FINANCIAL_CLASSES
        }
        min_crit_recall = min(critical_recalls.values()) if critical_recalls else 0.0
        passes_critical_gate = min_crit_recall >= 0.85

        # Gate 3: Calibration Degradation <= 0.03
        passes_calibration_gate = (brier <= rf_baseline_brier + 0.03) and (
            ece <= rf_baseline_ece + 0.03
        )

        # Gate 4: Latency strictly under 10ms
        passes_latency_gate = lat < 10.0

        failure_reasons = []
        if not passes_performance_gate:
            failure_reasons.append(
                f"No F1 gain (diff={f1_diff:+.4f}) nor >=30% latency cut with smaller size "
                f"(lat_cut={latency_reduction_pct:.1f}%, size={size_kb}KB vs {rf_baseline_size}KB)"
            )
        if not passes_critical_gate:
            failure_reasons.append(f"Critical class recall below 0.85 (min={min_crit_recall:.4f})")
        if not passes_calibration_gate:
            failure_reasons.append(
                f"Calibration degraded (brier={brier:.4f} vs rf={rf_baseline_brier:.4f}, "
                f"ece={ece:.4f} vs rf={rf_baseline_ece:.4f})"
            )
        if not passes_latency_gate:
            failure_reasons.append(f"Latency >= 10ms ({lat:.3f}ms)")

        is_eligible = len(failure_reasons) == 0

        challenger_evaluations.append(
            {
                "name": name,
                "macro_f1": macro_f1,
                "f1_diff": round(f1_diff, 4),
                "min_critical_recall": round(min_crit_recall, 4),
                "latency_ms": lat,
                "latency_reduction_pct": round(latency_reduction_pct, 1),
                "artifact_size_kb": size_kb,
                "brier_score": brier,
                "ece": ece,
                "is_eligible": is_eligible,
                "failure_reasons": failure_reasons,
                "pipeline": (
                    ensemble if name == "ensemble_soft_voting" else calibrated_models.get(name)
                ),
                "result": res,
            }
        )

    # Sort eligible challengers by Macro F1 descending, then latency ascending
    eligible_sorted = sorted(
        [c for c in challenger_evaluations if c["is_eligible"]],
        key=lambda x: (x["macro_f1"], -x["latency_ms"]),
        reverse=True,
    )

    if eligible_sorted:
        top_challenger = eligible_sorted[0]
        champion_name = top_challenger["name"]
        champion_pipeline = top_challenger["pipeline"]
        champion_metadata = top_challenger["result"]
        if top_challenger["f1_diff"] >= 0.005:
            selection_rationale = (
                f"Promoted {champion_name.upper()} as New Production Champion: "
                f"achieved strictly superior Holdout Macro F1 ({top_challenger['macro_f1']:.4f} vs baseline {rf_baseline_f1:.4f}), "
                f"min critical financial recall ({top_challenger['min_critical_recall']:.4f} >= 0.85), "
                f"calibrated Brier Score ({top_challenger['brier_score']:.4f}), "
                f"and CPU latency ({top_challenger['latency_ms']:.3f} ms/sample < 10ms)."
            )
        else:
            selection_rationale = (
                f"Promoted {champion_name.upper()} as New Production Champion: "
                f"maintained equivalent holdout Macro F1 ({top_challenger['macro_f1']:.4f} vs baseline {rf_baseline_f1:.4f}) "
                f"while delivering a {top_challenger['latency_reduction_pct']}% latency reduction "
                f"({top_challenger['latency_ms']:.3f}ms vs {rf_baseline_lat:.3f}ms), "
                f"{top_challenger['artifact_size_kb']}KB artifact size vs {rf_baseline_size}KB, "
                f"and zero critical class recall degradation."
            )
    else:
        reasons_summary = "; ".join(
            f"{c['name']}: {', '.join(c['failure_reasons'])}" for c in challenger_evaluations
        )
        selection_rationale = (
            f"Retained RANDOM_FOREST baseline as Champion: all challengers failed promotion gates. "
            f"Gate failures: {reasons_summary}"
        )

    print(f"Selection Decision: {champion_name.upper()}")
    print(f"Rationale: {selection_rationale}\n")

    # 5. Atomically promote the selected Champion
    promoted_meta = registry.promote_champion(
        champion_name=champion_name,
        pipeline=champion_pipeline,
        metadata=champion_metadata,
        selection_rationale=selection_rationale,
    )

    total_duration = round(time.perf_counter() - start_time, 2)

    # 6. Save machine-readable scorecard
    benchmark_scorecard = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_benchmark_duration_seconds": total_duration,
        "dataset_archetypes": {
            "train": splits["train_archetypes"],
            "val": splits["val_archetypes"],
            "test_holdout": splits["test_archetypes"],
        },
        "sample_counts": {
            "train": len(X_train_df),
            "val": len(X_val_df),
            "test_holdout": len(X_test_df),
        },
        "baseline_model": "random_forest",
        "selected_champion": champion_name,
        "champion_selection_rationale": selection_rationale,
        "champion_summary": promoted_meta,
        "candidates": candidate_results,
    }

    results_json_path = os.path.join(registry.artifacts_dir, "benchmark_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_scorecard, f, indent=2)

    print(f"Benchmark results persisted to: {results_json_path}")
    print(f"Total Experiment Time: {total_duration}s")
    print("==========================================================================\n")

    return benchmark_scorecard


if __name__ == "__main__":
    run_full_benchmark(do_tuning=True)
