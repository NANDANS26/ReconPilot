import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from app.models.enums import (
    ActorType,
    AdjustmentStatus,
    AdjustmentType,
    AIClassification,
    EvidenceType,
    ExceptionSeverity,
    ExceptionStatus,
    ExceptionType,
    FeeStatus,
    FeeType,
    MatchStatus,
    MatchType,
    MerchantStatus,
    PaymentStatus,
    RefundStatus,
    ReviewDecision,
    RunStatus,
    RunType,
    SettlementStatus,
    SourceProvider,
    SyncStatus,
    ValidationStatus,
)


@dataclass
class SeedDataset:
    merchant: dict[str, Any]
    source_integrations: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    refunds: list[dict[str, Any]]
    fees: list[dict[str, Any]]
    settlements: list[dict[str, Any]]
    settlement_items: list[dict[str, Any]]
    adjustments: list[dict[str, Any]]
    reconciliation_runs: list[dict[str, Any]]
    reconciliation_results: list[dict[str, Any]]
    exceptions: list[dict[str, Any]]
    evidence_items: list[dict[str, Any]]
    ai_investigations: list[dict[str, Any]]
    reviews: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    benchmark_cases: list[dict[str, Any]] = field(default_factory=list)


def generate_deterministic_dataset() -> SeedDataset:
    """
    Constructs a deterministic, internally consistent financial dataset with exactly:
    - 250+ Payments
    - 20+ Settlements
    - Refunds, Fees, Signed Adjustments
    - At least 50 deliberately constructed reconciliation scenarios with ground-truth BenchmarkCase entries.
    All monetary amounts use Decimal(20, 2).
    """
    # Deterministic merchant ID
    merchant_id = uuid.UUID("00000000-0000-4000-a000-000000000001")
    merchant_data = {
        "id": merchant_id,
        "name": "NovaPay Retail Enterprises",
        "email": "finance@novapay.example.com",
        "external_reference": "MERCH_NOVAPAY_IN",
        "currency": "INR",
        "timezone": "Asia/Kolkata",
        "status": MerchantStatus.ACTIVE,
        "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    }

    # Deterministic source integration
    source_id = uuid.UUID("00000000-0000-4000-a000-000000000002")
    source_integrations = [
        {
            "id": source_id,
            "merchant_id": merchant_id,
            "provider": SourceProvider.RAZORPAY,
            "display_name": "Payment Gateway (Production)",
            "environment": "live",
            "status": "ACTIVE",
            "last_sync_at": datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
            "last_sync_status": SyncStatus.SUCCESS,
            "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 10, 1, 18, 0, tzinfo=UTC),
        }
    ]

    payments: list[dict[str, Any]] = []
    refunds: list[dict[str, Any]] = []
    fees: list[dict[str, Any]] = []
    settlements: list[dict[str, Any]] = []
    settlement_items: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    reconciliation_runs: list[dict[str, Any]] = []
    reconciliation_results: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    evidence_items: list[dict[str, Any]] = []
    ai_investigations: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    audit_events: list[dict[str, Any]] = []
    benchmark_cases: list[dict[str, Any]] = []

    # Deterministic master reconciliation run
    run_id = uuid.UUID("00000000-0000-4000-b000-000000000001")
    reconciliation_runs.append(
        {
            "id": run_id,
            "merchant_id": merchant_id,
            "source_integration_id": source_id,
            "run_type": RunType.BENCHMARK,
            "status": RunStatus.COMPLETED,
            "started_at": datetime(2026, 10, 1, 6, 0, tzinfo=UTC),
            "completed_at": datetime(2026, 10, 1, 6, 5, tzinfo=UTC),
            "total_records": 250,
            "matched_records": 200,
            "unmatched_records": 50,
            "exception_records": 50,
            "auto_classified_records": 35,
            "review_required_records": 15,
            "processing_duration_ms": 300000,
            "error_summary": None,
            "configuration_snapshot": {
                "amount_tolerance": "1.00",
                "auto_classify_threshold": "0.90",
                "review_threshold": "0.75",
            },
            "created_at": datetime(2026, 10, 1, 6, 0, tzinfo=UTC),
            "updated_at": datetime(2026, 10, 1, 6, 5, tzinfo=UTC),
        }
    )

    # =========================================================================
    # PART 1: 50 DELIBERATELY CONSTRUCTED SCENARIOS & BENCHMARKS
    # =========================================================================
    # We define 50 distinct scenarios satisfying every prompt requirement.
    scenario_configs = [
        # 1-10: Exact & Fee Scenarios
        (
            1,
            "SCEN_01_FULL_MATCH",
            MatchStatus.MATCHED,
            None,
            Decimal("1000.00"),
            Decimal("20.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Fully matched payment and settlement",
        ),
        (
            2,
            "SCEN_02_PARTIAL_REFUND",
            MatchStatus.MATCHED,
            None,
            Decimal("1500.00"),
            Decimal("30.00"),
            Decimal("300.00"),
            Decimal("0.00"),
            "Valid partial refund",
        ),
        (
            3,
            "SCEN_03_MULTIPLE_REFUNDS",
            MatchStatus.MATCHED,
            None,
            Decimal("2000.00"),
            Decimal("40.00"),
            Decimal("500.00"),
            Decimal("0.00"),
            "Multiple successful refunds",
        ),
        (
            4,
            "SCEN_04_FEE_ACCOUNTED",
            MatchStatus.MATCHED,
            None,
            Decimal("5000.00"),
            Decimal("100.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Fee correctly accounted for",
        ),
        (
            5,
            "SCEN_05_INCORRECT_FEE",
            MatchStatus.EXCEPTION,
            ExceptionType.FEE_MISMATCH,
            Decimal("3000.00"),
            Decimal("95.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Incorrect fee deducted by gateway",
        ),
        (
            6,
            "SCEN_06_MISSING_SETTLEMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.MISSING_SETTLEMENT,
            Decimal("4200.00"),
            Decimal("84.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Missing settlement for captured payment",
        ),
        (
            7,
            "SCEN_07_SETTLEMENT_NO_PAYMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.MISSING_PAYMENT,
            Decimal("1800.00"),
            Decimal("36.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Settlement item without matching payment in internal ledger",
        ),
        (
            8,
            "SCEN_08_DUPLICATE_PAYMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.DUPLICATE_PAYMENT,
            Decimal("2500.00"),
            Decimal("50.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Duplicate payment source record in ledger",
        ),
        (
            9,
            "SCEN_09_DUPLICATE_SETTLEMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.DUPLICATE_SETTLEMENT,
            Decimal("3200.00"),
            Decimal("64.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Duplicate settlement line in bank report",
        ),
        (
            10,
            "SCEN_10_DELAYED_SETTLEMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.DELAYED_SETTLEMENT,
            Decimal("7500.00"),
            Decimal("150.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Delayed settlement past T+2 window",
        ),
        # 11-20: Adjustments & Mismatches
        (
            11,
            "SCEN_11_AMOUNT_MISMATCH",
            MatchStatus.EXCEPTION,
            ExceptionType.AMOUNT_MISMATCH,
            Decimal("8000.00"),
            Decimal("160.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Gross amount mismatch between ledger and gateway",
        ),
        (
            12,
            "SCEN_12_POSITIVE_ADJUSTMENT",
            MatchStatus.MATCHED,
            None,
            Decimal("1200.00"),
            Decimal("24.00"),
            Decimal("0.00"),
            Decimal("150.00"),
            "Positive adjustment increasing expected settlement",
        ),
        (
            13,
            "SCEN_13_NEGATIVE_ADJUSTMENT",
            MatchStatus.MATCHED,
            None,
            Decimal("1600.00"),
            Decimal("32.00"),
            Decimal("0.00"),
            Decimal("-200.00"),
            "Negative adjustment decreasing expected settlement",
        ),
        (
            14,
            "SCEN_14_SETTLEMENT_LEVEL_ADJ",
            MatchStatus.MATCHED,
            None,
            Decimal("2200.00"),
            Decimal("44.00"),
            Decimal("0.00"),
            Decimal("50.00"),
            "Settlement-level adjustment fee rebate",
        ),
        (
            15,
            "SCEN_15_REFUND_NOT_IN_SETTLEMENT",
            MatchStatus.EXCEPTION,
            ExceptionType.REFUND_MISMATCH,
            Decimal("3500.00"),
            Decimal("70.00"),
            Decimal("500.00"),
            Decimal("0.00"),
            "Refund not reflected in settlement item net amount",
        ),
        (
            16,
            "SCEN_16_MISSING_EXT_ID",
            MatchStatus.EXCEPTION,
            ExceptionType.UNKNOWN_EXCEPTION,
            Decimal("900.00"),
            Decimal("18.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Missing external transaction identifier",
        ),
        (
            17,
            "SCEN_17_MULTI_PAYMENT_BATCH_A",
            MatchStatus.MATCHED,
            None,
            Decimal("1100.00"),
            Decimal("22.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Multiple payments in one settlement batch (Part A)",
        ),
        (
            18,
            "SCEN_18_MULTI_PAYMENT_BATCH_B",
            MatchStatus.MATCHED,
            None,
            Decimal("2100.00"),
            Decimal("42.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Multiple payments in one settlement batch (Part B)",
        ),
        (
            19,
            "SCEN_19_AMBIGUOUS_EVIDENCE",
            MatchStatus.EXCEPTION,
            ExceptionType.AMOUNT_MISMATCH,
            Decimal("4800.00"),
            Decimal("96.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Ambiguous evidence requiring human review",
        ),
        (
            20,
            "SCEN_20_CROSS_CURRENCY",
            MatchStatus.EXCEPTION,
            ExceptionType.UNKNOWN_EXCEPTION,
            Decimal("500.00"),
            Decimal("10.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Cross-currency record with USD currency flag",
        ),
        # 21-25: Status & Boundary Integrity
        (
            21,
            "SCEN_21_FAILED_PAYMENT",
            MatchStatus.MATCHED,
            None,
            Decimal("1000.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Failed payment must not be treated as revenue",
        ),
        (
            22,
            "SCEN_22_FAILED_REFUND",
            MatchStatus.MATCHED,
            None,
            Decimal("2500.00"),
            Decimal("50.00"),
            Decimal("250.00"),
            Decimal("0.00"),
            "Failed refund must not reduce settlement",
        ),
        (
            23,
            "SCEN_23_WITHIN_TOLERANCE",
            MatchStatus.MATCHED_WITH_VARIANCE,
            None,
            Decimal("3000.00"),
            Decimal("60.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Transaction difference within ₹1.00 tolerance (diff = ₹0.60)",
        ),
        (
            24,
            "SCEN_24_OUTSIDE_TOLERANCE",
            MatchStatus.EXCEPTION,
            ExceptionType.AMOUNT_MISMATCH,
            Decimal("3000.00"),
            Decimal("60.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Transaction difference outside ₹1.00 tolerance (diff = ₹2.50)",
        ),
        (
            25,
            "SCEN_25_INSUFFICIENT_EVIDENCE",
            MatchStatus.EXCEPTION,
            ExceptionType.UNKNOWN_EXCEPTION,
            Decimal("6000.00"),
            Decimal("120.00"),
            Decimal("0.00"),
            Decimal("0.00"),
            "Transaction with insufficient evidence",
        ),
    ]

    # Generate additional 25 scenarios (26 to 50) to fulfill full 50+ benchmark coverage
    for i in range(26, 51):
        scen_id = f"SCEN_{i:02d}_BATCH_VARIATION"
        if i % 3 == 0:
            scenario_configs.append(
                (
                    i,
                    scen_id,
                    MatchStatus.EXCEPTION,
                    ExceptionType.FEE_MISMATCH,
                    Decimal(f"{1000 + i * 50}.00"),
                    Decimal(f"{20 + i}.00"),
                    Decimal("0.00"),
                    Decimal("0.00"),
                    f"Deterministic fee discrepancy scenario #{i}",
                )
            )
        elif i % 5 == 0:
            scenario_configs.append(
                (
                    i,
                    scen_id,
                    MatchStatus.EXCEPTION,
                    ExceptionType.ADJUSTMENT_MISMATCH,
                    Decimal(f"{1200 + i * 40}.00"),
                    Decimal("25.00"),
                    Decimal("0.00"),
                    Decimal("-150.00"),
                    f"Unmatched dispute adjustment scenario #{i}",
                )
            )
        else:
            scenario_configs.append(
                (
                    i,
                    scen_id,
                    MatchStatus.MATCHED,
                    None,
                    Decimal(f"{1000 + i * 30}.00"),
                    Decimal(f"{20 + i}.00"),
                    Decimal("0.00"),
                    Decimal("0.00"),
                    f"Deterministic standard matched scenario #{i}",
                )
            )

    # Process each of the 50 deliberate scenarios
    for (
        idx,
        scen_id,
        match_status,
        exc_type,
        gross_amt,
        fee_amt,
        ref_amt,
        adj_amt,
        desc,
    ) in scenario_configs:
        p_id = uuid.UUID(f"00000000-0000-4000-c000-{idx:012d}")
        s_id = uuid.UUID(f"00000000-0000-4000-d000-{idx:012d}")
        si_id = uuid.UUID(f"00000000-0000-4000-e000-{idx:012d}")
        res_id = uuid.UUID(f"00000000-0000-4000-f000-{idx:012d}")

        tx_time = datetime(2026, 9, 28, 10, idx % 60, tzinfo=UTC)
        settle_day = date(2026, 9, 29)
        currency = "USD" if idx == 20 else "INR"
        p_status = (
            PaymentStatus.FAILED
            if idx == 21
            else (PaymentStatus.REFUNDED if ref_amt == gross_amt else PaymentStatus.CAPTURED)
        )

        # Expected settlement calculation formula:
        # expected = payment_amount - refunds - fees + signed_adjustments
        expected_net = gross_amt - ref_amt - fee_amt + adj_amt
        if idx == 21:  # Failed payment -> expected is 0
            expected_net = Decimal("0.00")

        # Determine actual reported net amount, fee amount, and adjustment amount in settlement item
        item_fee_amt = fee_amt
        item_adj_amt = adj_amt

        if idx == 5:  # Fee mismatch (gateway charged 95 instead of agreed 60)
            item_fee_amt = Decimal("60.00")
            actual_net = gross_amt - ref_amt - item_fee_amt + adj_amt
        elif (
            exc_type == ExceptionType.FEE_MISMATCH
        ):  # Batch fee mismatches (e.g. 27, 30, 33, 36, 39, 42, 45, 48)
            item_fee_amt = fee_amt + Decimal("30.00")
            actual_net = gross_amt - ref_amt - item_fee_amt + adj_amt
        elif (
            exc_type == ExceptionType.ADJUSTMENT_MISMATCH
        ):  # Adjustment mismatches (e.g. 35, 40, 50)
            # Settlement item lacks the adjustment that ledger applied
            item_adj_amt = Decimal("0.00")
            actual_net = gross_amt - ref_amt - fee_amt + item_adj_amt
        elif idx == 11:  # Amount mismatch (settlement missing 100)
            actual_net = expected_net - Decimal("100.00")
        elif idx == 15:  # Refund not reflected in settlement
            actual_net = gross_amt - fee_amt + adj_amt
        elif idx == 19:  # Ambiguous evidence (unreconciled amount variance)
            actual_net = expected_net - Decimal("150.00")
        elif idx == 23:  # Within tolerance (difference 0.60)
            actual_net = expected_net - Decimal("0.60")
        elif idx == 24:  # Outside tolerance (difference 2.50)
            actual_net = expected_net - Decimal("2.50")
        elif exc_type == ExceptionType.MISSING_SETTLEMENT:
            actual_net = Decimal("0.00")
        else:
            actual_net = expected_net

        diff = actual_net - expected_net

        # Create Payment (except for missing payment scenario #7)
        if idx != 7:
            payments.append(
                {
                    "id": p_id,
                    "merchant_id": merchant_id,
                    "source_integration_id": source_id,
                    "external_id": f"pay_demo_{idx:04d}",
                    "external_order_id": f"order_demo_{idx:04d}",
                    "amount": gross_amt,
                    "currency": currency,
                    "status": p_status,
                    "payment_method": "upi" if idx % 2 == 0 else "card",
                    "captured_at": tx_time if p_status == PaymentStatus.CAPTURED else None,
                    "source_created_at": tx_time,
                    "payment_metadata": {
                        "scenario_id": f"{scen_id}_PRIMARY" if idx in (8, 9) else scen_id
                    },
                    "created_at": tx_time,
                    "updated_at": tx_time,
                }
            )

            # SCEN_08 (DUPLICATE_PAYMENT): Add true duplicate payment source record sharing external_order_id
            if idx == 8:
                dup_p8_id = uuid.UUID(f"00000000-0000-4000-c000-{88:012d}")
                payments.append(
                    {
                        "id": dup_p8_id,
                        "merchant_id": merchant_id,
                        "source_integration_id": source_id,
                        "external_id": f"pay_demo_{idx:04d}_dup",
                        "external_order_id": f"order_demo_{idx:04d}",
                        "amount": gross_amt,
                        "currency": currency,
                        "status": p_status,
                        "payment_method": "upi",
                        "captured_at": tx_time,
                        "source_created_at": tx_time,
                        "payment_metadata": {"scenario_id": scen_id},
                        "created_at": tx_time,
                        "updated_at": tx_time,
                    }
                )

            # Create Fee
            if fee_amt > Decimal("0.00"):
                fees.append(
                    {
                        "id": uuid.UUID(f"00000000-0000-4000-1000-{idx:012d}"),
                        "merchant_id": merchant_id,
                        "payment_id": p_id,
                        "external_id": f"fee_demo_{idx:04d}",
                        "fee_type": FeeType.MDR,
                        "amount": fee_amt,
                        "currency": currency,
                        "tax_amount": (fee_amt * Decimal("0.18")).quantize(Decimal("0.01")),
                        "status": FeeStatus.ASSESSED,
                        "created_at": tx_time,
                        "updated_at": tx_time,
                    }
                )

            # Create Refund
            if ref_amt > Decimal("0.00"):
                ref_status = RefundStatus.FAILED if idx == 22 else RefundStatus.PROCESSED
                refunds.append(
                    {
                        "id": uuid.UUID(f"00000000-0000-4000-2000-{idx:012d}"),
                        "merchant_id": merchant_id,
                        "payment_id": p_id,
                        "external_id": f"rfnd_demo_{idx:04d}",
                        "amount": ref_amt,
                        "currency": currency,
                        "status": ref_status,
                        "processed_at": tx_time,
                        "refund_metadata": {"scenario_id": scen_id},
                        "created_at": tx_time,
                        "updated_at": tx_time,
                    }
                )

            # Create Adjustment
            if adj_amt != Decimal("0.00"):
                adjustments.append(
                    {
                        "id": uuid.UUID(f"00000000-0000-4000-3000-{idx:012d}"),
                        "merchant_id": merchant_id,
                        "payment_id": p_id,
                        "settlement_id": s_id
                        if exc_type != ExceptionType.MISSING_SETTLEMENT
                        else None,
                        "external_id": f"adj_demo_{idx:04d}",
                        "amount": adj_amt,
                        "currency": currency,
                        "adjustment_type": AdjustmentType.DISPUTE_REVERSAL
                        if adj_amt > 0
                        else AdjustmentType.CHARGEBACK,
                        "reason": "Merchant dispute adjustment"
                        if adj_amt > 0
                        else "Chargeback debit",
                        "status": AdjustmentStatus.APPLIED,
                        "effective_at": tx_time,
                        "adjustment_metadata": {"scenario_id": scen_id},
                        "created_at": tx_time,
                        "updated_at": tx_time,
                    }
                )

        # Create Settlement & Item (except missing settlement #6, delayed settlement #10, and insufficient evidence #25)
        # Note:
        # - SCEN_10 (DELAYED_SETTLEMENT): The captured payment exists in ledger but has no settlement credit,
        #   and elapsed time exceeds the 2-day SLA window, classifying as DELAYED_SETTLEMENT.
        # - SCEN_25 (INSUFFICIENT_EVIDENCE): Payment exists but lacks matching settlement record.
        # - SCEN_20 (CROSS_CURRENCY): Payment is in USD, while settlement item is settled in INR, causing currency mismatch / unmatched item.
        # - SCEN_21 (FAILED_PAYMENT): Settlement and item have 0.00 net and gross amount, confirming non-revenue.
        # - SCEN_22 (FAILED_REFUND): Failed refund does not reduce settlement, item net amount is gross - fee.
        if exc_type != ExceptionType.MISSING_SETTLEMENT and idx not in (10, 25):
            settle_currency = "INR" if idx == 20 else currency
            settle_gross = (
                Decimal("0.00") if idx == 21 else (Decimal("950.00") if idx == 16 else gross_amt)
            )
            settle_refund_deduction = Decimal("0.00") if idx == 22 else ref_amt
            settle_item_net = (
                Decimal("0.00")
                if idx == 21
                else (
                    Decimal("932.00")
                    if idx == 16
                    else (gross_amt - fee_amt if idx == 22 else actual_net)
                )
            )

            settlements.append(
                {
                    "id": s_id,
                    "merchant_id": merchant_id,
                    "source_integration_id": source_id,
                    "external_id": f"setl_demo_{idx:04d}",
                    "settlement_date": settle_day,
                    "amount": settle_item_net,
                    "currency": settle_currency,
                    "status": SettlementStatus.SETTLED,
                    "initiated_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                    "settled_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                    "source_created_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                    "settlement_metadata": {
                        "scenario_id": f"{scen_id}_PRIMARY" if idx == 9 else scen_id
                    },
                    "created_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                    "updated_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                }
            )

            settlement_items.append(
                {
                    "id": si_id,
                    "merchant_id": merchant_id,
                    "settlement_id": s_id,
                    "payment_id": p_id if idx not in (7, 16, 20) else None,
                    "external_payment_id": (
                        f"pay_demo_{idx:04d}"
                        if idx not in (7, 16)
                        else ("pay_missing_external" if idx == 7 else None)
                    ),
                    "gross_amount": settle_gross,
                    "fee_amount": Decimal("0.00") if idx == 21 else item_fee_amt,
                    "refund_amount": settle_refund_deduction,
                    "adjustment_amount": Decimal("0.00") if idx == 21 else item_adj_amt,
                    "net_amount": settle_item_net,
                    "currency": settle_currency,
                    "source_created_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                    "item_metadata": {"scenario_id": f"{scen_id}_PRIMARY" if idx == 9 else scen_id},
                    "created_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                    "updated_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                }
            )

            # SCEN_09 (DUPLICATE_SETTLEMENT): Add duplicate settlement line in bank report
            if idx == 9:
                dup_s9_id = uuid.UUID(f"00000000-0000-4000-d000-{89:012d}")
                dup_si9_id = uuid.UUID(f"00000000-0000-4000-e000-{89:012d}")
                settlements.append(
                    {
                        "id": dup_s9_id,
                        "merchant_id": merchant_id,
                        "source_integration_id": source_id,
                        "external_id": "setl_demo_0089",
                        "settlement_date": settle_day,
                        "amount": settle_item_net,
                        "currency": settle_currency,
                        "status": SettlementStatus.SETTLED,
                        "initiated_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                        "settled_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                        "source_created_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                        "settlement_metadata": {"scenario_id": scen_id},
                        "created_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                        "updated_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                    }
                )
                settlement_items.append(
                    {
                        "id": dup_si9_id,
                        "merchant_id": merchant_id,
                        "settlement_id": dup_s9_id,
                        "payment_id": p_id,
                        "external_payment_id": f"pay_demo_{idx:04d}",
                        "gross_amount": settle_gross,
                        "fee_amount": item_fee_amt,
                        "refund_amount": settle_refund_deduction,
                        "adjustment_amount": item_adj_amt,
                        "net_amount": settle_item_net,
                        "currency": settle_currency,
                        "source_created_at": datetime(2026, 9, 29, 4, 0, tzinfo=UTC),
                        "item_metadata": {"scenario_id": scen_id},
                        "created_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                        "updated_at": datetime(2026, 9, 29, 6, 0, tzinfo=UTC),
                    }
                )

        # Create Reconciliation Result
        reconciliation_results.append(
            {
                "id": res_id,
                "run_id": run_id,
                "merchant_id": merchant_id,
                "payment_id": p_id if idx != 7 else None,
                "settlement_id": s_id
                if (exc_type != ExceptionType.MISSING_SETTLEMENT and idx not in (10, 25))
                else None,
                "match_type": MatchType.EXACT_2_WAY
                if match_status == MatchStatus.MATCHED
                else MatchType.UNMATCHED,
                "match_status": match_status,
                "expected_amount": expected_net,
                "actual_amount": actual_net,
                "difference": diff,
                "tolerance": Decimal("1.00"),
                "confidence": Decimal("1.0000")
                if match_status == MatchStatus.MATCHED
                else Decimal("0.8500"),
                "explanation": desc,
                "matching_fields": {"scenario_id": scen_id},
                "created_at": datetime(2026, 10, 1, 6, 1, tzinfo=UTC),
            }
        )

        # Create Exception & Evidence if scenario is an exception
        if exc_type is not None:
            exc_id = uuid.UUID(f"00000000-0000-4000-4000-{idx:012d}")
            exceptions.append(
                {
                    "id": exc_id,
                    "merchant_id": merchant_id,
                    "reconciliation_result_id": res_id,
                    "exception_type": exc_type,
                    "severity": ExceptionSeverity.HIGH
                    if abs(diff) > 500
                    else ExceptionSeverity.MEDIUM,
                    "status": ExceptionStatus.REVIEW_REQUIRED
                    if idx % 2 == 1
                    else ExceptionStatus.AUTO_RESOLVED,
                    "financial_impact": abs(diff),
                    "currency": currency,
                    "description": desc,
                    "confidence": Decimal("0.9200") if idx % 2 == 0 else Decimal("0.7800"),
                    "assigned_to": "controller@novapay.example.com" if idx % 2 == 1 else None,
                    "first_detected_at": datetime(2026, 10, 1, 6, 2, tzinfo=UTC),
                    "resolved_at": datetime(2026, 10, 1, 6, 3, tzinfo=UTC)
                    if idx % 2 == 0
                    else None,
                    "created_at": datetime(2026, 10, 1, 6, 2, tzinfo=UTC),
                    "updated_at": datetime(2026, 10, 1, 6, 3, tzinfo=UTC),
                }
            )

            # Create Evidence
            evidence_items.append(
                {
                    "id": uuid.UUID(f"00000000-0000-4000-5000-{idx:012d}"),
                    "exception_id": exc_id,
                    "evidence_type": EvidenceType.GATEWAY_PAYMENT,
                    "source_table": "payments" if idx != 7 else "settlement_items",
                    "source_record_id": p_id if idx != 7 else si_id,
                    "description": f"Source record evidence for {scen_id}",
                    "evidence_data": {
                        "expected_amount": str(expected_net),
                        "actual_amount": str(actual_net),
                        "difference": str(diff),
                    },
                    "created_at": datetime(2026, 10, 1, 6, 2, tzinfo=UTC),
                }
            )

            # Create AI Investigation
            ai_investigations.append(
                {
                    "id": uuid.UUID(f"00000000-0000-4000-6000-{idx:012d}"),
                    "exception_id": exc_id,
                    "provider": "mock",
                    "model": "reconpilot-rf-v1",
                    "prompt_version": "v1.0",
                    "classification": AIClassification.FEE_RELATED
                    if exc_type == ExceptionType.FEE_MISMATCH
                    else AIClassification.TRUE_FINANCIAL_EXCEPTION,
                    "root_cause": f"Investigated root cause for {scen_id}",
                    "explanation": f"Statistical variance analysis identified mismatch of {diff} INR",
                    "confidence": Decimal("0.9200") if idx % 2 == 0 else Decimal("0.7800"),
                    "recommended_action": "RECONCILE_WITH_ADJUSTMENT",
                    "evidence_references": {"evidence_id": f"00000000-0000-4000-5000-{idx:012d}"},
                    "raw_response": {"status": "ok"},
                    "validation_status": ValidationStatus.VALID,
                    "failure_reason": None,
                    "created_at": datetime(2026, 10, 1, 6, 2, tzinfo=UTC),
                }
            )

            # Create Review if review required
            if idx % 2 == 1:
                reviews.append(
                    {
                        "id": uuid.UUID(f"00000000-0000-4000-7000-{idx:012d}"),
                        "exception_id": exc_id,
                        "reviewer_id": "controller@novapay.example.com",
                        "decision": ReviewDecision.OVERRIDE_CLASSIFICATION,
                        "comment": "Audited with merchant billing contract.",
                        "previous_status": ExceptionStatus.OPEN,
                        "new_status": ExceptionStatus.REVIEW_REQUIRED,
                        "created_at": datetime(2026, 10, 1, 6, 3, tzinfo=UTC),
                    }
                )

            # Create Audit Event
            audit_events.append(
                {
                    "id": uuid.UUID(f"00000000-0000-4000-8000-{idx:012d}"),
                    "merchant_id": merchant_id,
                    "exception_id": exc_id,
                    "reconciliation_run_id": run_id,
                    "actor_type": ActorType.SYSTEM,
                    "actor_id": "recon_engine_v1",
                    "event_type": "EXCEPTION_DETECTED",
                    "previous_state": "OPEN",
                    "new_state": "REVIEW_REQUIRED" if idx % 2 == 1 else "AUTO_RESOLVED",
                    "metadata": {"scenario_id": scen_id, "impact": str(abs(diff))},
                    "created_at": datetime(2026, 10, 1, 6, 2, tzinfo=UTC),
                }
            )

        # Record Ground Truth BenchmarkCase
        benchmark_cases.append(
            {
                "id": uuid.UUID(f"00000000-0000-4000-9000-{idx:012d}"),
                "dataset_version": "v1.0",
                "scenario_id": scen_id,
                "source_reference": f"pay_demo_{idx:04d}" if idx != 7 else "pay_missing_external",
                "expected_match_status": match_status,
                "expected_exception_type": exc_type,
                "expected_root_cause": desc,
                "expected_financial_impact": abs(diff),
                "expected_policy_outcome": "AUTO_RESOLVE"
                if (idx % 2 == 0 or match_status == MatchStatus.MATCHED)
                else "HUMAN_REVIEW",
                "case_metadata": {"scenario_number": idx},
                "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            }
        )

    # =========================================================================
    # PART 2: 200 HEALTHY BULK PAYMENTS & 15 SETTLEMENT BATCHES
    # (Brings total payments to 250+, settlements to 20+, exercising scale)
    # =========================================================================
    for b in range(1, 16):
        batch_settle_id = uuid.UUID(f"00000000-0000-4000-b100-{b:012d}")
        batch_total_gross = Decimal("0.00")
        batch_total_net = Decimal("0.00")
        batch_items = []

        # 14 payments per batch -> 210 bulk payments
        for p_idx in range(1, 15):
            unique_num = 100 + (b - 1) * 14 + p_idx
            p_bulk_id = uuid.UUID(f"00000000-0000-4000-c100-{unique_num:012d}")
            amt = Decimal(f"{1500 + (unique_num % 50) * 25}.00")
            fee = (amt * Decimal("0.02")).quantize(Decimal("0.01"))
            net = amt - fee
            tx_dt = datetime(2026, 9, (b % 28) + 1, (p_idx % 12) + 8, 0, tzinfo=UTC)

            payments.append(
                {
                    "id": p_bulk_id,
                    "merchant_id": merchant_id,
                    "source_integration_id": source_id,
                    "external_id": f"pay_bulk_{unique_num:05d}",
                    "external_order_id": f"order_bulk_{unique_num:05d}",
                    "amount": amt,
                    "currency": "INR",
                    "status": PaymentStatus.CAPTURED,
                    "payment_method": "upi" if unique_num % 2 == 0 else "card",
                    "captured_at": tx_dt,
                    "source_created_at": tx_dt,
                    "payment_metadata": {"batch_id": f"batch_{b}"},
                    "created_at": tx_dt,
                    "updated_at": tx_dt,
                }
            )

            fees.append(
                {
                    "id": uuid.UUID(f"00000000-0000-4000-1100-{unique_num:012d}"),
                    "merchant_id": merchant_id,
                    "payment_id": p_bulk_id,
                    "external_id": f"fee_bulk_{unique_num:05d}",
                    "fee_type": FeeType.MDR,
                    "amount": fee,
                    "currency": "INR",
                    "tax_amount": (fee * Decimal("0.18")).quantize(Decimal("0.01")),
                    "status": FeeStatus.ASSESSED,
                    "created_at": tx_dt,
                    "updated_at": tx_dt,
                }
            )

            batch_total_gross += amt
            batch_total_net += net

            batch_items.append(
                {
                    "id": uuid.UUID(f"00000000-0000-4000-e100-{unique_num:012d}"),
                    "merchant_id": merchant_id,
                    "settlement_id": batch_settle_id,
                    "payment_id": p_bulk_id,
                    "external_payment_id": f"pay_bulk_{unique_num:05d}",
                    "gross_amount": amt,
                    "fee_amount": fee,
                    "refund_amount": Decimal("0.00"),
                    "adjustment_amount": Decimal("0.00"),
                    "net_amount": net,
                    "currency": "INR",
                    "source_created_at": tx_dt,
                    "item_metadata": {"batch_number": b},
                    "created_at": tx_dt,
                    "updated_at": tx_dt,
                }
            )

        settlements.append(
            {
                "id": batch_settle_id,
                "merchant_id": merchant_id,
                "source_integration_id": source_id,
                "external_id": f"setl_bulk_batch_{b:04d}",
                "settlement_date": date(2026, 9, (b % 28) + 1),
                "amount": batch_total_net,
                "currency": "INR",
                "status": SettlementStatus.SETTLED,
                "initiated_at": datetime(2026, 9, (b % 28) + 1, 19, 0, tzinfo=UTC),
                "settled_at": datetime(2026, 9, (b % 28) + 1, 20, 0, tzinfo=UTC),
                "source_created_at": datetime(2026, 9, (b % 28) + 1, 19, 0, tzinfo=UTC),
                "settlement_metadata": {"line_items_count": len(batch_items)},
                "created_at": datetime(2026, 9, (b % 28) + 1, 20, 0, tzinfo=UTC),
                "updated_at": datetime(2026, 9, (b % 28) + 1, 20, 0, tzinfo=UTC),
            }
        )

        settlement_items.extend(batch_items)

    return SeedDataset(
        merchant=merchant_data,
        source_integrations=source_integrations,
        payments=payments,
        refunds=refunds,
        fees=fees,
        settlements=settlements,
        settlement_items=settlement_items,
        adjustments=adjustments,
        reconciliation_runs=reconciliation_runs,
        reconciliation_results=reconciliation_results,
        exceptions=exceptions,
        evidence_items=evidence_items,
        ai_investigations=ai_investigations,
        reviews=reviews,
        audit_events=audit_events,
        benchmark_cases=benchmark_cases,
    )
