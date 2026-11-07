from decimal import Decimal
from typing import Any, NamedTuple

from app.models.enums import (
    AdjustmentStatus,
    FeeStatus,
    PaymentStatus,
    RefundStatus,
)
from app.modules.reconciliation.constants import DEFAULT_AMOUNT_TOLERANCE
from app.modules.reconciliation.normalization import (
    NormalizedAdjustment,
    NormalizedFee,
    NormalizedPayment,
    NormalizedRefund,
    NormalizedSettlementItem,
)


class CalculationBreakdown(NamedTuple):
    original_payment_amount: Decimal
    successful_refund_total: Decimal
    applicable_fee_total: Decimal
    applicable_adjustment_total: Decimal
    expected_settlement: Decimal
    actual_settlement: Decimal
    difference: Decimal
    currency: str
    applied_tolerance: Decimal
    is_within_tolerance: bool
    explanation: str


def calculate_expected_settlement_for_payment(
    payment: NormalizedPayment,
    refunds: list[NormalizedRefund],
    fees: list[NormalizedFee],
    adjustments: list[NormalizedAdjustment],
    actual_item: NormalizedSettlementItem | None = None,
    tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
) -> CalculationBreakdown:
    """
    Deterministic calculation of expected settlement amount for a captured payment.

    Formula:
    Expected Settlement =
        captured_payment_amount
        - sum(successful_refunds)
        - sum(applicable_fees, tax included exactly once)
        + sum(signed_applicable_adjustments)

    Convention:
    difference = expected_settlement - actual_settlement
    - Positive difference means actual settlement is lower than expected (under-settled / missing funds).
    - Negative difference means actual settlement is higher than expected (over-settled).
    """
    currency = payment.currency

    # Currency safety check
    for r in refunds:
        if r.currency != currency:
            raise ValueError(
                f"Refund currency {r.currency} does not match payment currency {currency}"
            )
    for f in fees:
        if f.currency != currency:
            raise ValueError(
                f"Fee currency {f.currency} does not match payment currency {currency}"
            )
    for a in adjustments:
        if a.currency != currency:
            raise ValueError(
                f"Adjustment currency {a.currency} does not match payment currency {currency}"
            )
    if actual_item and actual_item.currency != currency:
        raise ValueError(
            f"Settlement item currency {actual_item.currency} does not match payment currency {currency}"
        )

    # Deduplicate component records to prevent double-counting
    seen_refund_keys: set[Any] = set()
    deduped_refunds: list[NormalizedRefund] = []
    for r in refunds:
        r_ext = getattr(r, "external_id", None)
        key = (r.id, r_ext) if r_ext else r.id
        if key not in seen_refund_keys:
            seen_refund_keys.add(key)
            deduped_refunds.append(r)

    seen_fee_keys: set[Any] = set()
    deduped_fees: list[NormalizedFee] = []
    for f in fees:
        f_ext = getattr(f, "external_id", None)
        key = (f.id, f_ext) if f_ext else f.id
        if key not in seen_fee_keys:
            seen_fee_keys.add(key)
            deduped_fees.append(f)

    seen_adj_keys: set[Any] = set()
    deduped_adjustments: list[NormalizedAdjustment] = []
    for a in adjustments:
        a_ext = getattr(a, "external_id", None)
        key = (a.id, a_ext) if a_ext else a.id
        if key not in seen_adj_keys:
            seen_adj_keys.add(key)
            deduped_adjustments.append(a)

    # Non-captured payments (failed, authorized-only) do not produce captured revenue
    payment_amount = Decimal("0.00") if payment.status != PaymentStatus.CAPTURED else payment.amount

    # Only PROCESSED refunds reduce expected settlement; PENDING/FAILED refunds do not
    successful_refunds = [r for r in deduped_refunds if r.status == RefundStatus.PROCESSED]
    successful_refund_total = sum((r.amount for r in successful_refunds), Decimal("0.00"))

    # Only ASSESSED fees are deducted; WAIVED fees are excluded
    # fee.amount is the recorded fee deducted from gross settlement
    applicable_fees = [f for f in deduped_fees if f.status == FeeStatus.ASSESSED]
    applicable_fee_total = sum((f.amount for f in applicable_fees), Decimal("0.00"))

    # Only APPLIED adjustments affect expected settlement
    # Adjustments are signed: positive increases expected settlement, negative decreases
    applicable_adjustments = [
        a for a in deduped_adjustments if a.status == AdjustmentStatus.APPLIED
    ]
    applicable_adjustment_total = sum((a.amount for a in applicable_adjustments), Decimal("0.00"))

    # Calculate expected settlement
    expected_settlement = (
        payment_amount
        - successful_refund_total
        - applicable_fee_total
        + applicable_adjustment_total
    )

    actual_settlement = actual_item.net_amount if actual_item is not None else Decimal("0.00")
    difference = expected_settlement - actual_settlement
    is_within_tolerance = abs(difference) <= tolerance

    explanation = (
        f"Captured payment {payment.amount} {currency} - refunds ({successful_refund_total}) "
        f"- fees+tax ({applicable_fee_total}) + adjustments ({applicable_adjustment_total}) "
        f"= expected {expected_settlement}. Actual = {actual_settlement}. Diff = {difference}."
    )

    return CalculationBreakdown(
        original_payment_amount=payment_amount,
        successful_refund_total=successful_refund_total,
        applicable_fee_total=applicable_fee_total,
        applicable_adjustment_total=applicable_adjustment_total,
        expected_settlement=expected_settlement,
        actual_settlement=actual_settlement,
        difference=difference,
        currency=currency,
        applied_tolerance=tolerance,
        is_within_tolerance=is_within_tolerance,
        explanation=explanation,
    )
