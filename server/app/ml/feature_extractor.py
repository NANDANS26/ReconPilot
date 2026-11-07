"""
Feature Extractor for ReconPilot ML Investigation Engine.
Extracts purely objective, non-leaking features from an ExceptionEvidenceResponse
and its constituent financial records.
All financial arithmetic uses Decimal and integer minor units.
"""

import contextlib
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.ml.feature_schema import validate_feature_dict
from app.models.enums import EvidenceCategory, EvidenceStrength
from app.modules.evidence.schemas import ExceptionEvidenceResponse


def to_minor_units(amount: Decimal | float | None) -> int:
    """Converts a monetary amount to integer minor units (paise/cents)."""
    if amount is None:
        return 0
    try:
        dec = Decimal(str(amount))
        return int((dec * Decimal("100")).quantize(Decimal("1")))
    except Exception:
        return 0


class FeatureExtractor:
    """
    Extracts structured feature vector from an authoritative evidence dossier.
    Strictly isolated from ground-truth answer labels or entity UUID shortcuts.
    """

    @classmethod
    def extract_from_dossier(cls, dossier: ExceptionEvidenceResponse) -> dict[str, Any]:
        """
        Converts an ExceptionEvidenceResponse into a validated feature dictionary.
        """
        recon = dossier.reconciliation
        summary = dossier.summary
        related = dossier.related_records
        items = dossier.evidence_items

        # Tally evidence items by category and strength
        cat_counts: dict[EvidenceCategory, int] = {}
        conflict_count = 0
        incomplete_count = 0
        for item in items:
            cat_counts[item.category] = cat_counts.get(item.category, 0) + 1
            if item.strength == EvidenceStrength.CONFLICTING:
                conflict_count += 1
            elif item.strength == EvidenceStrength.INCOMPLETE:
                incomplete_count += 1

        has_payment = (
            1.0 if (related.payment_ids or cat_counts.get(EvidenceCategory.PAYMENT, 0) > 0) else 0.0
        )
        has_settle_item = (
            1.0
            if (related.settlement_item_ids or cat_counts.get(EvidenceCategory.SETTLEMENT, 0) > 0)
            else 0.0
        )
        has_settle_batch = (
            1.0 if (related.settlement_ids or len(related.settlement_ids) > 0) else 0.0
        )

        payment_minor = to_minor_units(summary.expected_amount) if has_payment else 0
        settlement_gross_minor = to_minor_units(summary.actual_amount) if has_settle_item else 0
        settlement_net_minor = to_minor_units(summary.actual_amount) if has_settle_item else 0

        expected_minor = to_minor_units(recon.expected_amount)
        actual_minor = to_minor_units(recon.actual_amount)
        variance_minor = to_minor_units(recon.variance)
        abs_variance_minor = abs(variance_minor)

        # Variance components from breakdown attributes if present
        fee_diff_minor = 0
        refund_diff_minor = 0
        adj_diff_minor = 0
        days_between = 0.0

        payment_timestamp: datetime | None = None
        settlement_timestamp: datetime | None = None

        for item in items:
            attrs = item.attributes or {}
            if item.category == EvidenceCategory.FEE:
                fee_diff_minor += to_minor_units(attrs.get("fee_amount", 0))
            elif item.category == EvidenceCategory.REFUND:
                refund_diff_minor += to_minor_units(attrs.get("refund_amount", 0))
            elif item.category == EvidenceCategory.ADJUSTMENT:
                adj_diff_minor += to_minor_units(attrs.get("amount", 0))

            if item.category == EvidenceCategory.PAYMENT and "captured_at" in attrs:
                with contextlib.suppress(Exception):
                    payment_timestamp = datetime.fromisoformat(str(attrs["captured_at"]))
            elif item.category == EvidenceCategory.SETTLEMENT and "settlement_date" in attrs:
                with contextlib.suppress(Exception):
                    settlement_timestamp = datetime.fromisoformat(str(attrs["settlement_date"]))

        if payment_timestamp and settlement_timestamp:
            try:
                days_between = float(
                    abs((settlement_timestamp - payment_timestamp).total_seconds()) / 86400.0
                )
            except Exception:
                days_between = 0.0

        # Duplicate counts
        dup_payment_count = (
            max(0, len(related.payment_ids) - 1) if len(related.payment_ids) > 1 else 0
        )
        dup_settle_count = (
            max(0, len(related.settlement_item_ids) - 1)
            if len(related.settlement_item_ids) > 1
            else 0
        )

        # Discrepancy / ambiguity indicators
        has_internal_id = (
            1.0 if (len(related.payment_ids) > 0 and len(related.settlement_item_ids) > 0) else 0.0
        )
        has_ext_id = 1.0 if has_payment and has_settle_item else 0.0
        is_curr_consistent = 1.0
        is_merch_consistent = 1.0
        is_ambiguous = (
            1.0 if len(summary.conflicts) > 0 and (has_payment and has_settle_item) else 0.0
        )

        raw_features: dict[str, Any] = {
            "has_payment": has_payment,
            "has_settlement_item": has_settle_item,
            "has_settlement_batch": has_settle_batch,
            "payment_amount_minor": payment_minor,
            "settlement_gross_minor": settlement_gross_minor,
            "settlement_net_minor": settlement_net_minor,
            "expected_amount_minor": expected_minor,
            "actual_amount_minor": actual_minor,
            "variance_minor": variance_minor,
            "abs_variance_minor": abs_variance_minor,
            "fee_difference_minor": fee_diff_minor,
            "refund_difference_minor": refund_diff_minor,
            "adjustment_difference_minor": adj_diff_minor,
            "fee_count": len(related.fee_ids),
            "refund_count": len(related.refund_ids),
            "adjustment_count": len(related.adjustment_ids),
            "candidate_count": max(1, len(related.payment_ids) + len(related.settlement_item_ids)),
            "duplicate_payment_count": dup_payment_count,
            "duplicate_settlement_count": dup_settle_count,
            "days_between_payment_and_settlement": days_between,
            "supporting_evidence_count": len(items),
            "contradicting_evidence_count": conflict_count,
            "missing_info_count": len(summary.missing_information),
            "conflict_count": len(summary.conflicts),
            "has_internal_id_match": has_internal_id,
            "has_external_id_match": has_ext_id,
            "is_currency_consistent": is_curr_consistent,
            "is_merchant_consistent": is_merch_consistent,
            "is_ambiguous": is_ambiguous,
            "has_incomplete_evidence": 1.0 if incomplete_count > 0 else 0.0,
            "has_conflicting_evidence": 1.0 if conflict_count > 0 else 0.0,
            "currency": recon.currency or "INR",
            "payment_status": "CAPTURED" if has_payment else "MISSING",
            "settlement_status": "SETTLED" if has_settle_item else "MISSING",
            "match_type": (
                recon.match_type.value
                if hasattr(recon, "match_type") and hasattr(recon.match_type, "value")
                else str(getattr(recon, "match_type", "EXACT_2_WAY"))
            ),
            "match_status": recon.match_status.value
            if hasattr(recon.match_status, "value")
            else str(recon.match_status),
        }

        return validate_feature_dict(raw_features)
