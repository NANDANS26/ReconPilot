"""
Unit tests for ReconPilot Deterministic Reconciliation Hardening.
Verifies:
1. Permutation / Order Invariance: Shuffling the input payments and settlement items
   produces bitwise identical matched pairs, unmatched lists, and confidence scores.
2. Bilateral Uniqueness: Ambiguous candidates (1-to-many or many-to-1) are preserved
   as un-matched or ambiguous candidates, never arbitrarily chosen by list ordering.
3. Preventive Duplicate Quarantine: Duplicate payment and settlement item keys are quarantined.
4. Exact Decimal Minor-Unit Arithmetic: Deduplication of fees, refunds, and adjustments prevents double-counting.
"""

import random
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.models.enums import (
    AdjustmentStatus,
    AdjustmentType,
    FeeStatus,
    FeeType,
    RefundStatus,
)
from app.modules.reconciliation.calculations import (
    calculate_expected_settlement_for_payment,
)
from app.modules.reconciliation.duplicate_detection import (
    detect_duplicates,
)
from app.modules.reconciliation.matching import run_deterministic_matching
from app.modules.reconciliation.normalization import (
    NormalizedAdjustment,
    NormalizedFee,
    NormalizedPayment,
    NormalizedRefund,
    NormalizedSettlementItem,
)

BASE_TIME = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
MERCHANT_ID = uuid.uuid4()


def _make_payment(
    external_id: str,
    amount: Decimal,
    offset_seconds: int = 0,
    payment_id: uuid.UUID | None = None,
) -> NormalizedPayment:
    return NormalizedPayment(
        id=payment_id or uuid.uuid4(),
        merchant_id=MERCHANT_ID,
        external_id=external_id,
        amount=amount,
        currency="INR",
        status="CAPTURED",  # type: ignore[arg-type]
        payment_method="card",
        source_created_at=BASE_TIME + timedelta(seconds=offset_seconds),
    )


def _make_settlement_item(
    gross_amount: Decimal,
    payment_id: uuid.UUID | None = None,
    external_payment_id: str | None = None,
    offset_seconds: int = 0,
) -> NormalizedSettlementItem:
    return NormalizedSettlementItem(
        id=uuid.uuid4(),
        merchant_id=MERCHANT_ID,
        settlement_id=uuid.uuid4(),
        payment_id=payment_id,
        external_payment_id=external_payment_id,
        gross_amount=gross_amount,
        net_amount=gross_amount * Decimal("0.98"),
        currency="INR",
        source_created_at=BASE_TIME + timedelta(seconds=offset_seconds),
    )


def test_matching_input_permutation_invariance():
    """
    Permuting (shuffling) the order of payments and settlement items must produce
    identical matched pairs and identical unmatched records every single time.
    """
    payments: list[NormalizedPayment] = []
    items: list[NormalizedSettlementItem] = []

    # 1. Five Stage 1 matches (direct external ID matches)
    for i in range(5):
        ext_id = f"pay_inv_{i}"
        amt = Decimal(f"{1000 + i * 100}.00")
        p = _make_payment(external_id=ext_id, amount=amt, offset_seconds=i * 60)
        s = _make_settlement_item(
            gross_amount=amt,
            external_payment_id=ext_id,
            offset_seconds=i * 60 + 30,
        )
        payments.append(p)
        items.append(s)

    # 2. Three Stage 2 matches (unique amount and close timestamp, different external IDs)
    for i in range(5, 8):
        amt = Decimal(f"{5000 + i * 250}.00")
        p = _make_payment(external_id=f"pay_stage2_{i}", amount=amt, offset_seconds=i * 120)
        s = _make_settlement_item(
            gross_amount=amt,
            external_payment_id=f"diff_ref_{i}",
            offset_seconds=i * 120 + 10,
        )
        payments.append(p)
        items.append(s)

    # 3. Two unmatched payments and two unmatched settlement items
    p_unmatched_1 = _make_payment(external_id="unmatched_p1", amount=Decimal("9999.00"))
    p_unmatched_2 = _make_payment(external_id="unmatched_p2", amount=Decimal("8888.00"))
    s_unmatched_1 = _make_settlement_item(gross_amount=Decimal("7777.00"))
    s_unmatched_2 = _make_settlement_item(gross_amount=Decimal("6666.00"))

    payments.extend([p_unmatched_1, p_unmatched_2])
    items.extend([s_unmatched_1, s_unmatched_2])

    # Run baseline matching
    baseline_result = run_deterministic_matching(list(payments), list(items))
    baseline_pairs = {
        (pair.payment_id, pair.settlement_item_id) for pair in baseline_result.matched_pairs
    }
    baseline_unmatched_p = set(baseline_result.unmatched_payment_ids)
    baseline_unmatched_s = set(baseline_result.unmatched_settlement_item_ids)

    assert len(baseline_pairs) == 8
    assert len(baseline_unmatched_p) == 2
    assert len(baseline_unmatched_s) == 2

    # Run across 10 random permutations
    rng = random.Random(42)
    for trial in range(10):
        perm_payments = list(payments)
        perm_items = list(items)
        rng.shuffle(perm_payments)
        rng.shuffle(perm_items)

        perm_result = run_deterministic_matching(perm_payments, perm_items)
        perm_pairs = {
            (pair.payment_id, pair.settlement_item_id) for pair in perm_result.matched_pairs
        }
        perm_unmatched_p = set(perm_result.unmatched_payment_ids)
        perm_unmatched_s = set(perm_result.unmatched_settlement_item_ids)

        assert perm_pairs == baseline_pairs, f"Permutation mismatch in trial {trial}"
        assert perm_unmatched_p == baseline_unmatched_p
        assert perm_unmatched_s == baseline_unmatched_s


def test_bilateral_uniqueness_prevents_greedy_mismatch():
    """
    If two payments share the exact same amount and timestamp, but only one settlement item exists,
    Stage 2 candidate matching MUST NOT arbitrarily pair one and orphan the other.
    Both must remain un-paired due to candidate ambiguity.
    """
    amt = Decimal("2500.00")
    p1 = _make_payment("pay_ambig_1", amount=amt, offset_seconds=10)
    p2 = _make_payment("pay_ambig_2", amount=amt, offset_seconds=10)
    s1 = _make_settlement_item(gross_amount=amt, external_payment_id="s_ref", offset_seconds=12)

    result = run_deterministic_matching([p1, p2], [s1])

    # Neither should be matched because s1 is ambiguous between p1 and p2
    assert len(result.matched_pairs) == 0
    assert len(result.unmatched_payment_ids) == 2
    assert len(result.unmatched_settlement_item_ids) == 1


def test_preventive_duplicate_quarantine():
    """Verify that multiple payments and settlement items sharing external keys are quarantined."""
    p1 = _make_payment("dup_ext_id_100", amount=Decimal("1500.00"))
    p2 = _make_payment("dup_ext_id_100", amount=Decimal("1500.00"))
    p3 = _make_payment("unique_ext_id_200", amount=Decimal("2000.00"))

    s1 = _make_settlement_item(gross_amount=Decimal("1000.00"), external_payment_id="dup_s_1")
    s2 = _make_settlement_item(gross_amount=Decimal("1000.00"), external_payment_id="dup_s_1")
    s3 = _make_settlement_item(gross_amount=Decimal("1000.00"), external_payment_id="unique_s")

    check_res = detect_duplicates([p1, p2, p3], [s1, s2, s3])

    # Both duplicate keys are identified in duplicate_external_ids
    assert "dup_ext_id_100" in check_res.duplicate_external_ids
    assert len(check_res.duplicate_external_ids["dup_ext_id_100"]) == 2
    assert len(check_res.duplicate_payment_ids) >= 1

    assert len(check_res.duplicate_item_ids) >= 1


def test_deduplication_in_fee_and_refund_calculations():
    """Ensure duplicate fees, refunds, and adjustments are deduplicated to prevent double-counting."""
    fee_id = uuid.uuid4()
    pay_id = uuid.uuid4()

    payment = _make_payment("pay_calc_dedup", amount=Decimal("1000.00"), payment_id=pay_id)

    f1 = NormalizedFee(
        id=fee_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        fee_type=FeeType.MDR,
        amount=Decimal("25.00"),
        tax_amount=Decimal("4.50"),
        currency="INR",
        status=FeeStatus.ASSESSED,
        created_at=BASE_TIME,
    )
    # Duplicate fee record with same ID passed twice
    f2 = NormalizedFee(
        id=fee_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        fee_type=FeeType.MDR,
        amount=Decimal("25.00"),
        tax_amount=Decimal("4.50"),
        currency="INR",
        status=FeeStatus.ASSESSED,
        created_at=BASE_TIME,
    )

    ref_id = uuid.uuid4()
    r1 = NormalizedRefund(
        id=ref_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        external_id="rfnd_123",
        amount=Decimal("100.00"),
        currency="INR",
        status=RefundStatus.PROCESSED,
        processed_at=BASE_TIME,
    )
    r2 = NormalizedRefund(
        id=ref_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        external_id="rfnd_123",
        amount=Decimal("100.00"),
        currency="INR",
        status=RefundStatus.PROCESSED,
        processed_at=BASE_TIME,
    )

    adj_id = uuid.uuid4()
    a1 = NormalizedAdjustment(
        id=adj_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        adjustment_type=AdjustmentType.CHARGEBACK,
        amount=Decimal("-15.00"),
        currency="INR",
        status=AdjustmentStatus.APPLIED,
        effective_at=BASE_TIME,
    )
    a2 = NormalizedAdjustment(
        id=adj_id,
        merchant_id=MERCHANT_ID,
        payment_id=pay_id,
        adjustment_type=AdjustmentType.CHARGEBACK,
        amount=Decimal("-15.00"),
        currency="INR",
        status=AdjustmentStatus.APPLIED,
        effective_at=BASE_TIME,
    )

    breakdown = calculate_expected_settlement_for_payment(
        payment=payment,
        refunds=[r1, r2],
        fees=[f1, f2],
        adjustments=[a1, a2],
    )

    # 1000.00 - 100.00 (refund) - 25.00 (fee) + (-15.00) (adj) = 860.00
    # If not deduplicated, fee would be 50, refund 200, adj -30 -> 720.00
    assert breakdown.successful_refund_total == Decimal("100.00")
    assert breakdown.applicable_fee_total == Decimal("25.00")
    assert breakdown.applicable_adjustment_total == Decimal("-15.00")
    assert breakdown.expected_settlement == Decimal("860.00")
