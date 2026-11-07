"""
Synthetic & Realistic Scenario Dataset Generator for ReconPilot ML Engine.
Generates realistic financial reconciliation scenarios across 12 distinct business archetypes.
Features:
- Realistic cross-category confounding factors and noise injection (eliminating trivial single-feature fingerprints)
- Group-based scenario isolation (unseen business archetypes in holdout test set)
- Continuous variance distributions, multi-issue transactions, and borderline ambiguity
- Strict schema adherence to feature schema v1.0
"""

import random
from typing import Any

from app.ml.feature_schema import TARGET_LABELS, validate_feature_dict

# 12 Distinct Business Model Archetypes for Group-Separated Partitioning
BUSINESS_ARCHETYPES = [
    # Training Archetypes (8)
    "RETAIL_ECOMMERCE",
    "SAAS_RECURRING",
    "FOOD_DELIVERY",
    "EDTECH_ANNUAL",
    "B2B_WHOLESALE",
    "HEALTHCARE_CLINIC",
    "UTILITY_BILLING",
    "LOGISTICS_COURIER",
    # Validation Archetypes (2)
    "GAMING_MICROTX",
    "FINTECH_NEOBANK",
    # Untouched Holdout Test Archetypes (2)
    "TRAVEL_AIRLINE",
    "CROSS_BORDER_EXPORT",
]

TRAIN_ARCHETYPES = set(BUSINESS_ARCHETYPES[:8])
VAL_ARCHETYPES = set(BUSINESS_ARCHETYPES[8:10])
TEST_ARCHETYPES = set(BUSINESS_ARCHETYPES[10:12])


class SyntheticFinancialDatasetGenerator:
    """
    Generates realistic, noise-injected reconciliation feature vectors across 11 target classes.
    Supports both legacy uniform generation and realistic group-separated scenario generation.
    """

    def __init__(self, seed: int = 42, realistic: bool = True):
        self.seed = seed
        self.rng = random.Random(seed)
        self.realistic = realistic

    def _get_archetype_params(self, archetype: str) -> dict[str, Any]:
        """Returns ticket size, fee rate, and timing profile for a business archetype."""
        profiles = {
            "RETAIL_ECOMMERCE": {
                "min_amt": 500,
                "max_amt": 15000,
                "fee_rate": 0.019,
                "refund_prob": 0.25,
                "days_mean": 2.0,
            },
            "SAAS_RECURRING": {
                "min_amt": 1000,
                "max_amt": 50000,
                "fee_rate": 0.022,
                "refund_prob": 0.05,
                "days_mean": 1.5,
            },
            "FOOD_DELIVERY": {
                "min_amt": 250,
                "max_amt": 2500,
                "fee_rate": 0.025,
                "refund_prob": 0.15,
                "days_mean": 1.0,
            },
            "EDTECH_ANNUAL": {
                "min_amt": 15000,
                "max_amt": 150000,
                "fee_rate": 0.018,
                "refund_prob": 0.08,
                "days_mean": 2.5,
            },
            "B2B_WHOLESALE": {
                "min_amt": 50000,
                "max_amt": 500000,
                "fee_rate": 0.012,
                "refund_prob": 0.02,
                "days_mean": 3.0,
            },
            "HEALTHCARE_CLINIC": {
                "min_amt": 1000,
                "max_amt": 30000,
                "fee_rate": 0.017,
                "refund_prob": 0.04,
                "days_mean": 2.0,
            },
            "UTILITY_BILLING": {
                "min_amt": 300,
                "max_amt": 8000,
                "fee_rate": 0.015,
                "refund_prob": 0.01,
                "days_mean": 1.5,
            },
            "LOGISTICS_COURIER": {
                "min_amt": 400,
                "max_amt": 6000,
                "fee_rate": 0.020,
                "refund_prob": 0.06,
                "days_mean": 2.2,
            },
            "GAMING_MICROTX": {
                "min_amt": 100,
                "max_amt": 1500,
                "fee_rate": 0.028,
                "refund_prob": 0.12,
                "days_mean": 1.0,
            },
            "FINTECH_NEOBANK": {
                "min_amt": 1000,
                "max_amt": 100000,
                "fee_rate": 0.014,
                "refund_prob": 0.03,
                "days_mean": 0.5,
            },
            "TRAVEL_AIRLINE": {
                "min_amt": 4000,
                "max_amt": 80000,
                "fee_rate": 0.021,
                "refund_prob": 0.18,
                "days_mean": 3.5,
            },
            "CROSS_BORDER_EXPORT": {
                "min_amt": 20000,
                "max_amt": 300000,
                "fee_rate": 0.035,
                "refund_prob": 0.05,
                "days_mean": 4.0,
            },
        }
        return profiles.get(archetype, profiles["RETAIL_ECOMMERCE"])

    def generate_sample(self, label: str, archetype: str | None = None) -> dict[str, Any]:
        """Generates a realistic sample matching the specified label and business archetype."""
        if not archetype:
            archetype = self.rng.choice(BUSINESS_ARCHETYPES)

        params = self._get_archetype_params(archetype)
        currency = (
            "USD"
            if archetype == "CROSS_BORDER_EXPORT"
            else ("EUR" if self.rng.random() < 0.15 else "INR")
        )

        # Base amount in integer minor units (paise/cents)
        base_amt = self.rng.randint(params["min_amt"], params["max_amt"]) * 100
        normal_fee_pct = params["fee_rate"] + self.rng.uniform(-0.002, 0.002)
        normal_fee = int(base_amt * normal_fee_pct)

        # Base settlement days with Gaussian variation
        days_between = max(0.2, round(self.rng.gauss(params["days_mean"], 0.8), 2))

        # Baseline matched features
        feat: dict[str, Any] = {
            "has_payment": 1.0,
            "has_settlement_item": 1.0,
            "has_settlement_batch": 1.0,
            "payment_amount_minor": base_amt,
            "settlement_gross_minor": base_amt,
            "settlement_net_minor": base_amt - normal_fee,
            "expected_amount_minor": base_amt - normal_fee,
            "actual_amount_minor": base_amt - normal_fee,
            "variance_minor": 0,
            "abs_variance_minor": 0,
            "fee_difference_minor": 0,
            "refund_difference_minor": 0,
            "adjustment_difference_minor": 0,
            "fee_count": 1,
            "refund_count": 0,
            "adjustment_count": 0,
            "candidate_count": 1,
            "duplicate_payment_count": 0,
            "duplicate_settlement_count": 0,
            "days_between_payment_and_settlement": days_between,
            "supporting_evidence_count": self.rng.randint(3, 6),
            "contradicting_evidence_count": 0,
            "missing_info_count": 0,
            "conflict_count": 0,
            "has_internal_id_match": 1.0,
            "has_external_id_match": 1.0,
            "is_currency_consistent": 1.0,
            "is_merchant_consistent": 1.0,
            "is_ambiguous": 0.0,
            "has_incomplete_evidence": 0.0,
            "has_conflicting_evidence": 0.0,
            "currency": currency,
            "payment_status": "CAPTURED",
            "settlement_status": "SETTLED",
            "match_type": "EXACT_2_WAY",
            "match_status": "MATCHED",
        }

        # In realistic mode, inject background noise present in real operations
        if self.realistic:
            # 1. Natural refund presence (partial return on unrelated issue)
            if self.rng.random() < params["refund_prob"]:
                incidental_refund = int(base_amt * self.rng.uniform(0.1, 0.4))
                feat["refund_count"] = 1
                if label != "REFUND_MISMATCH":
                    # Legitimate refund properly factored into expected calculation
                    feat["expected_amount_minor"] -= incidental_refund
                    feat["actual_amount_minor"] -= incidental_refund
                    feat["settlement_net_minor"] -= incidental_refund
                    # Minor rounding drift in refund calculation
                    if self.rng.random() < 0.20:
                        feat["refund_difference_minor"] = self.rng.randint(10, 60)

            # 2. Legitimate adjustment presence (promotional credit, goodwill)
            if self.rng.random() < 0.15:
                incidental_adj = self.rng.randint(50, 500)
                feat["adjustment_count"] = 1
                if label != "ADJUSTMENT_MISMATCH":
                    feat["expected_amount_minor"] += incidental_adj
                    feat["actual_amount_minor"] += incidental_adj
                    feat["settlement_net_minor"] += incidental_adj
                    # Incidental minor adjustment variance
                    if self.rng.random() < 0.20:
                        feat["adjustment_difference_minor"] = self.rng.randint(15, 75)

            # 3. Legitimate multiple fees (MDR + fixed transaction fee + GST)
            if self.rng.random() < 0.40:
                feat["fee_count"] = self.rng.randint(2, 4)
                if label != "FEE_MISMATCH" and self.rng.random() < 0.25:
                    # Minor fee rounding variance (not primary cause)
                    feat["fee_difference_minor"] = self.rng.randint(5, 50)

            # 4. Realistic variation in days between payment and settlement
            if days_between > 4.5 and label != "DELAYED_SETTLEMENT":
                days_between = round(self.rng.uniform(2.0, 4.0), 2)
                feat["days_between_payment_and_settlement"] = days_between

            # 5. Background documentation ambiguity or minor missing docs
            if self.rng.random() < 0.12 and label not in (
                "INSUFFICIENT_EVIDENCE",
                "MISSING_PAYMENT",
                "MISSING_SETTLEMENT",
            ):
                feat["missing_info_count"] = 1
                if self.rng.random() < 0.5:
                    feat["has_incomplete_evidence"] = 1.0

            # 6. Candidate contention in high volume retail
            if self.rng.random() < 0.10 and label != "AMBIGUOUS_MATCH":
                feat["candidate_count"] = self.rng.randint(2, 3)
                if self.rng.random() < 0.4:
                    feat["is_ambiguous"] = 1.0

            # 7. Failed retry attempts in gaming/retail
            if self.rng.random() < 0.08 and label not in (
                "DUPLICATE_PAYMENT",
                "DUPLICATE_SETTLEMENT",
            ):
                if self.rng.random() < 0.5:
                    feat["duplicate_payment_count"] = 1
                else:
                    feat["duplicate_settlement_count"] = 1
                feat["conflict_count"] = 1

        # Apply Category-Specific Logic with realistic overlaps and non-trivial boundaries
        if label == "FEE_MISMATCH":
            # Fee discrepancy: gateway deducted a different fee rate than scheduled
            unauthorized_pct = self.rng.uniform(0.002, 0.015)
            fee_drift = max(15, int(base_amt * unauthorized_pct))
            feat["fee_difference_minor"] = fee_drift
            feat["actual_amount_minor"] = feat["expected_amount_minor"] - fee_drift
            feat["variance_minor"] = -fee_drift
            feat["abs_variance_minor"] = fee_drift
            feat["match_status"] = "EXCEPTION"
            if self.realistic and self.rng.random() < 0.20:
                # Confounding: minor settlement delay also present
                feat["days_between_payment_and_settlement"] = round(self.rng.uniform(3.5, 6.0), 2)

        elif label == "REFUND_MISMATCH":
            # Discrepancy specifically caused by an un-credited or miscalculated refund
            unsettled_refund = max(35, int(base_amt * self.rng.uniform(0.05, 0.70)))
            feat["refund_difference_minor"] = unsettled_refund
            feat["refund_count"] = max(1, feat["refund_count"])
            feat["actual_amount_minor"] = feat["expected_amount_minor"] - unsettled_refund
            feat["variance_minor"] = -unsettled_refund
            feat["abs_variance_minor"] = unsettled_refund
            feat["match_status"] = "EXCEPTION"
            if self.realistic and self.rng.random() < 0.25:
                # Fee also varied slightly
                subtle_fee = self.rng.randint(10, 50)
                feat["fee_difference_minor"] = subtle_fee

        elif label == "ADJUSTMENT_MISMATCH":
            # Chargeback or dispute adjustment disagreement
            adj_drift = max(40, int(base_amt * self.rng.uniform(0.03, 0.35)))
            sign = self.rng.choice([1, -1])
            feat["adjustment_difference_minor"] = adj_drift
            feat["adjustment_count"] = max(1, feat["adjustment_count"])
            feat["actual_amount_minor"] = feat["expected_amount_minor"] + (sign * adj_drift)
            feat["variance_minor"] = sign * adj_drift
            feat["abs_variance_minor"] = adj_drift
            feat["match_status"] = "EXCEPTION"

        elif label == "AMOUNT_MISMATCH":
            # Discrepancy in base gross charge not explained by fee/refund/adj schedules
            gross_drift = max(200, int(base_amt * self.rng.uniform(0.05, 0.25)))
            sign = self.rng.choice([1, -1])
            feat["variance_minor"] = sign * gross_drift
            feat["abs_variance_minor"] = gross_drift
            feat["actual_amount_minor"] = feat["expected_amount_minor"] + (sign * gross_drift)
            feat["match_status"] = "EXCEPTION"
            # In realistic mode, fees might still have normal minor variation
            if self.realistic and self.rng.random() < 0.20:
                feat["fee_difference_minor"] = self.rng.randint(5, 30)

        elif label == "MISSING_PAYMENT":
            # Orphan settlement: money deposited, but no internal payment record
            feat["has_payment"] = 0.0
            feat["payment_amount_minor"] = 0
            feat["payment_status"] = "MISSING"
            feat["expected_amount_minor"] = 0
            feat["variance_minor"] = feat["actual_amount_minor"]
            feat["abs_variance_minor"] = feat["actual_amount_minor"]
            feat["has_internal_id_match"] = 0.0
            feat["has_external_id_match"] = 0.0
            feat["missing_info_count"] = self.rng.randint(1, 3)
            feat["match_status"] = "UNMATCHED"
            if self.realistic and self.rng.random() < 0.15:
                feat["candidate_count"] = 2  # Possible near-match

        elif label == "MISSING_SETTLEMENT":
            # Payment captured internally, but missing from settlement batches
            feat["has_settlement_item"] = 0.0
            feat["has_settlement_batch"] = 0.0
            feat["settlement_gross_minor"] = 0
            feat["settlement_net_minor"] = 0
            feat["actual_amount_minor"] = 0
            feat["settlement_status"] = "MISSING"
            feat["variance_minor"] = -feat["expected_amount_minor"]
            feat["abs_variance_minor"] = feat["expected_amount_minor"]
            feat["has_internal_id_match"] = 0.0
            feat["has_external_id_match"] = 0.0
            feat["missing_info_count"] = self.rng.randint(1, 3)
            feat["match_status"] = "UNMATCHED"
            # Note: Days elapsed might be moderate or high
            feat["days_between_payment_and_settlement"] = round(self.rng.uniform(3.0, 10.0), 2)

        elif label == "DUPLICATE_PAYMENT":
            # Multiple payments claim the same authorization or external payment ID
            dup_count = self.rng.randint(1, 3)
            feat["duplicate_payment_count"] = dup_count
            feat["conflict_count"] = dup_count
            feat["contradicting_evidence_count"] = dup_count
            feat["has_conflicting_evidence"] = 1.0
            feat["match_status"] = "EXCEPTION"
            if self.realistic and self.rng.random() < 0.20:
                feat["candidate_count"] = dup_count + 1

        elif label == "DUPLICATE_SETTLEMENT":
            # Multiple settlement lines deposit the same external transaction reference
            dup_count = self.rng.randint(1, 3)
            feat["duplicate_settlement_count"] = dup_count
            feat["conflict_count"] = dup_count
            feat["contradicting_evidence_count"] = dup_count
            feat["has_conflicting_evidence"] = 1.0
            feat["match_status"] = "EXCEPTION"

        elif label == "DELAYED_SETTLEMENT":
            # Settlement arrived significantly beyond normal SLA (T+5 to T+20)
            feat["days_between_payment_and_settlement"] = round(self.rng.uniform(5.5, 21.0), 2)
            feat["settlement_status"] = "PENDING" if self.rng.random() < 0.4 else "SETTLED"
            feat["match_status"] = "EXCEPTION"
            if self.realistic and self.rng.random() < 0.20:
                # May also have minor fee adjustment
                feat["fee_difference_minor"] = self.rng.randint(10, 80)

        elif label == "AMBIGUOUS_MATCH":
            # Multiple competing records share identical or near-identical amounts within SLA
            feat["candidate_count"] = self.rng.randint(2, 5)
            feat["is_ambiguous"] = 1.0
            feat["has_internal_id_match"] = 0.0
            feat["has_external_id_match"] = 0.0 if self.rng.random() < 0.7 else 1.0
            feat["match_type"] = "TOLERANCE_2_WAY"
            feat["match_status"] = "EXCEPTION"
            feat["conflict_count"] = self.rng.randint(1, 3)
            if self.realistic:
                # Variance can be 0 or small tolerance difference
                variance = self.rng.choice([0, self.rng.randint(10, 200)])
                feat["variance_minor"] = variance
                feat["abs_variance_minor"] = variance

        elif label == "INSUFFICIENT_EVIDENCE":
            # Dossier lacks required corroborating evidence to reach any conclusion
            feat["missing_info_count"] = self.rng.randint(2, 6)
            feat["has_incomplete_evidence"] = 1.0
            feat["supporting_evidence_count"] = self.rng.randint(0, 2)
            feat["match_status"] = "EXCEPTION"
            feat["has_internal_id_match"] = 0.0 if self.rng.random() < 0.8 else 1.0
            if self.realistic and self.rng.random() < 0.3:
                feat["conflict_count"] = 1

        return validate_feature_dict(feat)

    def generate_dataset(
        self,
        samples_per_class: int = 150,
        archetypes: set[str] | list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Generates a balanced dataset across all target categories using specified archetypes.
        Returns (features_list, labels_list).
        """
        features: list[dict[str, Any]] = []
        labels: list[str] = []

        active_archetypes = list(archetypes) if archetypes else BUSINESS_ARCHETYPES

        for label in TARGET_LABELS:
            for i in range(samples_per_class):
                arch = active_archetypes[i % len(active_archetypes)]
                sample = self.generate_sample(label, archetype=arch)
                features.append(sample)
                labels.append(label)

        # Deterministic shuffle
        combined = list(zip(features, labels, strict=False))
        self.rng.shuffle(combined)
        shuffled_features, shuffled_labels = zip(*combined, strict=False)

        return list(shuffled_features), list(shuffled_labels)


def generate_benchmark_splits(
    train_seed: int = 42,
    val_seed: int = 123,
    test_seed: int = 999,
    samples_per_class_train: int = 200,
    samples_per_class_val: int = 50,
    samples_per_class_test: int = 100,
) -> dict[str, Any]:
    """
    Generates strictly group-separated, independent datasets for training,
    hyperparameter validation, and untouched holdout benchmarking.
    - Training set: 8 distinct business archetypes
    - Validation set: 2 distinct business archetypes (tuning only)
    - Holdout test set: 2 completely unseen business archetypes + unseen seed (final test only)
    """
    print(f"Generating Training Data (Archetypes: {len(TRAIN_ARCHETYPES)}, seed={train_seed})...")
    gen_train = SyntheticFinancialDatasetGenerator(seed=train_seed, realistic=True)
    X_train_raw, y_train = gen_train.generate_dataset(
        samples_per_class=samples_per_class_train, archetypes=TRAIN_ARCHETYPES
    )

    print(f"Generating Validation Data (Archetypes: {len(VAL_ARCHETYPES)}, seed={val_seed})...")
    gen_val = SyntheticFinancialDatasetGenerator(seed=val_seed, realistic=True)
    X_val_raw, y_val = gen_val.generate_dataset(
        samples_per_class=samples_per_class_val, archetypes=VAL_ARCHETYPES
    )

    print(f"Generating Holdout Test Data (Archetypes: {len(TEST_ARCHETYPES)}, seed={test_seed})...")
    gen_test = SyntheticFinancialDatasetGenerator(seed=test_seed, realistic=True)
    X_test_raw, y_test = gen_test.generate_dataset(
        samples_per_class=samples_per_class_test, archetypes=TEST_ARCHETYPES
    )

    return {
        "train": (X_train_raw, y_train),
        "val": (X_val_raw, y_val),
        "test": (X_test_raw, y_test),
        "train_archetypes": sorted(list(TRAIN_ARCHETYPES)),
        "val_archetypes": sorted(list(VAL_ARCHETYPES)),
        "test_archetypes": sorted(list(TEST_ARCHETYPES)),
    }
