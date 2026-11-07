"""
Feature Schema and Contract for ReconPilot Financial Exception Classifier.
Version: v1.0
All numeric financial amounts must be converted from Decimal to integer minor units
(e.g., paise, cents) before model evaluation to prevent binary floating-point drift.
Target leakage is strictly forbidden: no exception codes, result IDs, or entity UUIDs.
"""

from typing import Any

FEATURE_SCHEMA_VERSION = "v1.0"

# Numeric features (converted to float for scikit-learn standard scaling / imputation)
NUMERIC_FEATURES = [
    # Transaction availability
    "has_payment",
    "has_settlement_item",
    "has_settlement_batch",
    # Monetary values in integer minor units (paise / cents)
    "payment_amount_minor",
    "settlement_gross_minor",
    "settlement_net_minor",
    "expected_amount_minor",
    "actual_amount_minor",
    "variance_minor",
    "abs_variance_minor",
    # Variance components in minor units
    "fee_difference_minor",
    "refund_difference_minor",
    "adjustment_difference_minor",
    # Record counts
    "fee_count",
    "refund_count",
    "adjustment_count",
    "candidate_count",
    "duplicate_payment_count",
    "duplicate_settlement_count",
    # Time delta
    "days_between_payment_and_settlement",
    # Evidence counts
    "supporting_evidence_count",
    "contradicting_evidence_count",
    "missing_info_count",
    "conflict_count",
    # Flags (0 or 1)
    "has_internal_id_match",
    "has_external_id_match",
    "is_currency_consistent",
    "is_merchant_consistent",
    "is_ambiguous",
    "has_incomplete_evidence",
    "has_conflicting_evidence",
]

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    "currency",
    "payment_status",
    "settlement_status",
    "match_type",
    "match_status",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target root-cause classes for multiclass classification
TARGET_LABELS = [
    "FEE_MISMATCH",
    "REFUND_MISMATCH",
    "ADJUSTMENT_MISMATCH",
    "AMOUNT_MISMATCH",
    "MISSING_PAYMENT",
    "MISSING_SETTLEMENT",
    "DUPLICATE_PAYMENT",
    "DUPLICATE_SETTLEMENT",
    "DELAYED_SETTLEMENT",
    "AMBIGUOUS_MATCH",
    "INSUFFICIENT_EVIDENCE",
]

# Core financial anomaly classes subject to strict recall threshold gates
CRITICAL_FINANCIAL_CLASSES = [
    "MISSING_SETTLEMENT",
    "DUPLICATE_PAYMENT",
    "AMOUNT_MISMATCH",
    "FEE_MISMATCH",
    "REFUND_MISMATCH",
]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TARGET_LABELS)}
INDEX_TO_LABEL = {idx: label for idx, label in enumerate(TARGET_LABELS)}


def validate_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Validates that a dictionary of extracted features conforms to the schema.
    Fills default values for missing expected keys safely.
    """
    validated: dict[str, Any] = {}

    for num_col in NUMERIC_FEATURES:
        val = features.get(num_col, 0.0)
        try:
            validated[num_col] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            validated[num_col] = 0.0

    for cat_col in CATEGORICAL_FEATURES:
        val = features.get(cat_col, "UNKNOWN")
        validated[cat_col] = str(val).upper() if val is not None else "UNKNOWN"

    return validated
