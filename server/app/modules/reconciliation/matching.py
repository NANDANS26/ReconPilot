"""
Deterministic Reconciliation Matching Module for ReconPilot.
Implements multi-stage, permutation-invariant, bilateral 1-to-1 matching.
Enforces merchant and currency isolation, strict candidate uniqueness, and explicit ambiguity detection.
"""

import uuid
from collections import defaultdict
from decimal import Decimal
from typing import NamedTuple

from app.models.enums import MatchType
from app.modules.reconciliation.constants import (
    STAGE1_CONFIDENCE,
    STAGE2_CONFIDENCE,
)
from app.modules.reconciliation.normalization import (
    NormalizedPayment,
    NormalizedSettlementItem,
)


class MatchCandidate(NamedTuple):
    payment_id: uuid.UUID
    settlement_item_id: uuid.UUID
    match_type: MatchType
    confidence: Decimal
    matching_fields: dict[str, str]
    explanation: str


class MatchingResult(NamedTuple):
    matched_pairs: list[MatchCandidate]
    unmatched_payment_ids: set[uuid.UUID]
    unmatched_settlement_item_ids: set[uuid.UUID]
    ambiguous_pairs: list[dict[str, str]]


def run_deterministic_matching(
    payments: list[NormalizedPayment],
    settlement_items: list[NormalizedSettlementItem],
    date_window_days: int = 7,
) -> MatchingResult:
    """
    Executes 3-stage deterministic reconciliation matching:
    Stage 1: Exact relationship matching (payment.id == item.payment_id OR payment.external_id == item.external_payment_id)
             with collision protection (if multiple items claim same payment, marks collision).
    Stage 2: Bilateral strong reference matching (merchant + currency + exact gross amount within date window).
             Order-invariant: candidate pair matched IF AND ONLY IF uniquely supported on both sides.
    Stage 3: Controlled ambiguity isolation (ambiguous candidates kept unresolved for human review).
    """
    matched_pairs: list[MatchCandidate] = []
    matched_payment_ids: set[uuid.UUID] = set()
    matched_item_ids: set[uuid.UUID] = set()
    ambiguous_pairs: list[dict[str, str]] = []

    # Fast lookup indexes
    payment_by_id = {p.id: p for p in payments}
    payment_by_ext_id = {p.external_id: p for p in payments if p.external_id}

    # =========================================================================
    # STAGE 1: Exact relationship matching with bilateral collision detection
    # =========================================================================
    # Collect candidate payments claimed by each item in Stage 1
    item_to_stage1_payment: dict[uuid.UUID, tuple[NormalizedPayment, str]] = {}
    payment_to_stage1_items: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)

    for item in settlement_items:
        target_payment: NormalizedPayment | None = None
        matched_rule = ""

        # 1.a Direct internal payment ID link
        if item.payment_id and item.payment_id in payment_by_id:
            candidate = payment_by_id[item.payment_id]
            if candidate.merchant_id == item.merchant_id and candidate.currency == item.currency:
                target_payment = candidate
                matched_rule = "internal_payment_id"

        # 1.b External payment ID link
        elif item.external_payment_id and item.external_payment_id in payment_by_ext_id:
            candidate = payment_by_ext_id[item.external_payment_id]
            if candidate.merchant_id == item.merchant_id and candidate.currency == item.currency:
                target_payment = candidate
                matched_rule = "external_payment_id"

        if target_payment:
            item_to_stage1_payment[item.id] = (target_payment, matched_rule)
            payment_to_stage1_items[target_payment.id].append(item.id)

    # Only match if exactly one item claims the payment in Stage 1
    for item_id, (target_payment, matched_rule) in item_to_stage1_payment.items():
        claiming_items = payment_to_stage1_items[target_payment.id]
        if len(claiming_items) == 1:
            matched_payment_ids.add(target_payment.id)
            matched_item_ids.add(item_id)
            matched_pairs.append(
                MatchCandidate(
                    payment_id=target_payment.id,
                    settlement_item_id=item_id,
                    match_type=MatchType.EXACT_2_WAY,
                    confidence=STAGE1_CONFIDENCE,
                    matching_fields={
                        "rule": matched_rule,
                        "merchant_id": str(target_payment.merchant_id),
                        "currency": target_payment.currency,
                        "external_payment_id": target_payment.external_id or "",
                    },
                    explanation=f"Exact match via {matched_rule}.",
                )
            )
        else:
            # Collision: multiple items claim same payment
            ambiguous_pairs.append(
                {
                    "payment_id": str(target_payment.id),
                    "candidate_settlement_item_ids": [str(it) for it in claiming_items],
                    "reason": f"Multiple ({len(claiming_items)}) settlement items claim payment {target_payment.id}",
                }
            )

    # Remaining unmatched pools
    remaining_payments = [p for p in payments if p.id not in matched_payment_ids]
    remaining_items = [item for item in settlement_items if item.id not in matched_item_ids]

    # =========================================================================
    # STAGE 2: Bilateral strong reference matching (order-invariant)
    # =========================================================================
    # Build bidirectional candidate maps between remaining items and remaining payments
    item_candidates: dict[uuid.UUID, list[NormalizedPayment]] = defaultdict(list)
    payment_candidates: dict[uuid.UUID, list[NormalizedSettlementItem]] = defaultdict(list)

    for item in remaining_items:
        for p in remaining_payments:
            if (
                p.merchant_id == item.merchant_id
                and p.currency == item.currency
                and p.amount == item.gross_amount
                and abs((p.source_created_at - item.source_created_at).total_seconds())
                <= date_window_days * 86400
            ):
                item_candidates[item.id].append(p)
                payment_candidates[p.id].append(item)

    # Sort items and payments by stable ID to guarantee identical deterministic iteration order
    sorted_remaining_items = sorted(remaining_items, key=lambda x: x.id)

    for item in sorted_remaining_items:
        if item.id in matched_item_ids:
            continue

        cands = item_candidates.get(item.id, [])
        if len(cands) == 1:
            candidate_p = cands[0]
            # Check bilateral uniqueness: item must also be the ONLY candidate for this payment
            p_cands = payment_candidates.get(candidate_p.id, [])
            if len(p_cands) == 1 and p_cands[0].id == item.id:
                matched_payment_ids.add(candidate_p.id)
                matched_item_ids.add(item.id)
                matched_pairs.append(
                    MatchCandidate(
                        payment_id=candidate_p.id,
                        settlement_item_id=item.id,
                        match_type=MatchType.TOLERANCE_2_WAY,
                        confidence=STAGE2_CONFIDENCE,
                        matching_fields={
                            "rule": "exact_amount_and_timestamp_window",
                            "merchant_id": str(item.merchant_id),
                            "currency": item.currency,
                            "amount": str(item.gross_amount),
                        },
                        explanation="Strong match on merchant, currency, amount, and close creation timestamps.",
                    )
                )
            elif len(p_cands) > 1:
                ambiguous_pairs.append(
                    {
                        "settlement_item_id": str(item.id),
                        "payment_id": str(candidate_p.id),
                        "reason": (
                            f"Payment {candidate_p.id} matches multiple ({len(p_cands)}) "
                            f"settlement items with amount {item.gross_amount} {item.currency}."
                        ),
                    }
                )
        elif len(cands) > 1:
            ambiguous_pairs.append(
                {
                    "settlement_item_id": str(item.id),
                    "candidate_payment_ids": [str(c.id) for c in cands],
                    "reason": (
                        f"Multiple ({len(cands)}) payments have exact same amount "
                        f"{item.gross_amount} {item.currency} within date window."
                    ),
                }
            )

    # =========================================================================
    # STAGE 3: Final candidate set calculation
    # =========================================================================
    unmatched_payment_ids = {p.id for p in payments if p.id not in matched_payment_ids}
    unmatched_settlement_item_ids = {
        item.id for item in settlement_items if item.id not in matched_item_ids
    }

    return MatchingResult(
        matched_pairs=matched_pairs,
        unmatched_payment_ids=unmatched_payment_ids,
        unmatched_settlement_item_ids=unmatched_settlement_item_ids,
        ambiguous_pairs=ambiguous_pairs,
    )
