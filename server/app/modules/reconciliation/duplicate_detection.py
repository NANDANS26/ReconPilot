"""
Duplicate Detection Module for ReconPilot Deterministic Reconciliation.
Identifies duplicate source records and external reference collisions.
Does NOT mistake legitimate multiple payments of identical amounts as duplicates.
Requires duplicate external_id or identical relationship keys to classify as true duplicates.
"""

import uuid
from collections import defaultdict
from typing import NamedTuple

from app.modules.reconciliation.normalization import (
    NormalizedAdjustment,
    NormalizedFee,
    NormalizedPayment,
    NormalizedRefund,
    NormalizedSettlementItem,
)


class DuplicateCheckResult(NamedTuple):
    # IDs of payments identified as duplicate copies
    duplicate_payment_ids: set[uuid.UUID]
    # IDs of settlement items identified as duplicate copies
    duplicate_item_ids: set[uuid.UUID]
    # IDs of duplicate fees, refunds, and adjustments
    duplicate_fee_ids: set[uuid.UUID]
    duplicate_refund_ids: set[uuid.UUID]
    duplicate_adjustment_ids: set[uuid.UUID]
    # Mapping of colliding external IDs to record IDs
    duplicate_external_ids: dict[str, list[uuid.UUID]]


def detect_duplicates(
    payments: list[NormalizedPayment],
    settlement_items: list[NormalizedSettlementItem],
    fees: list[NormalizedFee] | None = None,
    refunds: list[NormalizedRefund] | None = None,
    adjustments: list[NormalizedAdjustment] | None = None,
) -> DuplicateCheckResult:
    """
    Identifies duplicate source records and external reference collisions across all components.
    Guarantees that duplicate records are quarantined and do not silently inflate financial matches.
    """
    fees = fees or []
    refunds = refunds or []
    adjustments = adjustments or []

    # 1. Check duplicate external_ids or duplicate external_order_ids across payments
    payments_by_ext_id: dict[str, list[uuid.UUID]] = defaultdict(list)
    payments_by_order_id: dict[str, list[uuid.UUID]] = defaultdict(list)
    duplicate_payment_ids: set[uuid.UUID] = set()
    dup_ext_ids: dict[str, list[uuid.UUID]] = {}

    for p in payments:
        if p.external_id:
            payments_by_ext_id[p.external_id].append(p.id)
        if p.external_order_id:
            payments_by_order_id[p.external_order_id].append(p.id)

    for ext_id, p_ids in payments_by_ext_id.items():
        if len(p_ids) > 1:
            dup_ext_ids[ext_id] = p_ids
            # Quarantine all subsequent duplicate payments
            duplicate_payment_ids.update(p_ids[1:])

    for order_id, p_ids in payments_by_order_id.items():
        if len(p_ids) > 1:
            dup_ext_ids[f"order:{order_id}"] = p_ids
            duplicate_payment_ids.update(p_ids[1:])

    # 2. Check duplicate settlement line items referencing the same payment
    items_by_payment_id: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    items_by_ext_payment: dict[str, list[uuid.UUID]] = defaultdict(list)
    duplicate_item_ids: set[uuid.UUID] = set()

    for item in settlement_items:
        if item.payment_id:
            items_by_payment_id[item.payment_id].append(item.id)
        if item.external_payment_id:
            items_by_ext_payment[item.external_payment_id].append(item.id)

    for _p_id, item_ids in items_by_payment_id.items():
        if len(item_ids) > 1:
            duplicate_item_ids.update(item_ids[1:])

    for _ext_pay_id, item_ids in items_by_ext_payment.items():
        if len(item_ids) > 1:
            duplicate_item_ids.update(item_ids[1:])

    # 3. Check duplicate fees (by id and external_id if present)
    fees_by_ext_id: dict[str, list[uuid.UUID]] = defaultdict(list)
    duplicate_fee_ids: set[uuid.UUID] = set()
    for f in fees:
        f_ext = getattr(f, "external_id", None)
        if f_ext:
            fees_by_ext_id[f_ext].append(f.id)
    for _ext_id, f_ids in fees_by_ext_id.items():
        if len(f_ids) > 1:
            duplicate_fee_ids.update(f_ids[1:])

    # 4. Check duplicate refunds (by id and external_id if present)
    refunds_by_ext_id: dict[str, list[uuid.UUID]] = defaultdict(list)
    duplicate_refund_ids: set[uuid.UUID] = set()
    for r in refunds:
        r_ext = getattr(r, "external_id", None)
        if r_ext:
            refunds_by_ext_id[r_ext].append(r.id)
    for _ext_id, r_ids in refunds_by_ext_id.items():
        if len(r_ids) > 1:
            duplicate_refund_ids.update(r_ids[1:])

    # 5. Check duplicate adjustments (by id and external_id if present)
    adjustments_by_ext_id: dict[str, list[uuid.UUID]] = defaultdict(list)
    duplicate_adjustment_ids: set[uuid.UUID] = set()
    for a in adjustments:
        a_ext = getattr(a, "external_id", None)
        if a_ext:
            adjustments_by_ext_id[a_ext].append(a.id)
    for _ext_id, a_ids in adjustments_by_ext_id.items():
        if len(a_ids) > 1:
            duplicate_adjustment_ids.update(a_ids[1:])

    return DuplicateCheckResult(
        duplicate_payment_ids=duplicate_payment_ids,
        duplicate_item_ids=duplicate_item_ids,
        duplicate_fee_ids=duplicate_fee_ids,
        duplicate_refund_ids=duplicate_refund_ids,
        duplicate_adjustment_ids=duplicate_adjustment_ids,
        duplicate_external_ids=dup_ext_ids,
    )
