import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.seed import seed_database
from app.models.audit_event import AuditEvent
from app.models.enums import ActorType, PaymentStatus
from app.models.payment import Payment
from app.modules.reconciliation.schemas import (
    BatchIngestPaymentsRequest,
    BatchIngestResponse,
    BatchIngestResultItem,
    IngestionSummaryResponse,
    IngestPaymentItem,
)


def ingest_demo_financial_dataset(
    db: Session,
    merchant_id: uuid.UUID,
    actor_id: str | None = None,
) -> IngestionSummaryResponse:
    """
    Ingests or verifies the standardized demo financial records for a merchant.
    Idempotent: Uses existing source idempotency keys and constraints.
    """
    start_time = time.perf_counter()

    # Record INGESTION_STARTED audit event
    audit_start = AuditEvent(
        id=uuid.uuid4(),
        merchant_id=merchant_id,
        actor_type=ActorType.HUMAN_CONTROLLER if actor_id else ActorType.SYSTEM,
        actor_id=actor_id,
        event_type="INGESTION_STARTED",
        event_metadata={"source": "demo_synthetic_feed"},
    )
    db.add(audit_start)
    db.commit()

    # Run deterministic seeder (which checks existence before inserting)
    seed_result = seed_database(session=db, verbose=False)

    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # Record INGESTION_COMPLETED audit event
    audit_done = AuditEvent(
        id=uuid.uuid4(),
        merchant_id=merchant_id,
        actor_type=ActorType.HUMAN_CONTROLLER if actor_id else ActorType.SYSTEM,
        actor_id=actor_id,
        event_type="INGESTION_COMPLETED",
        event_metadata={"duration_ms": duration_ms, "result": seed_result},
    )
    db.add(audit_done)
    db.commit()

    newly_inserted = sum(seed_result.values())
    total_processed = (
        newly_inserted if newly_inserted > 0 else 259
    )  # Examined standard dataset batch

    return IngestionSummaryResponse(
        total_records_processed=total_processed,
        inserted_payments=seed_result.get("payments", 0),
        inserted_settlements=seed_result.get("settlements", 0),
        inserted_items=seed_result.get("settlement_items", 0),
        inserted_refunds=seed_result.get("refunds", 0),
        inserted_fees=seed_result.get("fees", 0),
        inserted_adjustments=seed_result.get("adjustments", 0),
        skipped_records=259 - newly_inserted if newly_inserted < 259 else 0,
        duration_ms=duration_ms,
    )


def ingest_payments_batch_idempotent(
    db: Session,
    merchant_id: uuid.UUID,
    payload: BatchIngestPaymentsRequest,
    actor_id: str | None = None,
) -> BatchIngestResponse:
    """
    Idempotently ingests a batch of payment records for a merchant.
    - If payment with external_id exists with identical amount & currency: marks IDEMPOTENT_SKIPPED.
    - If payment with external_id exists with mismatched amount & currency: marks PAYLOAD_CONFLICT.
    - If payment does not exist: inserts into database with INSERTED status.
    - Enforces merchant isolation and audits batch ingestion.
    """
    results: list[BatchIngestResultItem] = []
    inserted_count = 0
    idempotent_skipped_count = 0
    conflict_count = 0

    # Query all existing payments for this merchant matching the external_ids in the batch
    incoming_external_ids = [p.external_id for p in payload.payments]
    existing_stmt = select(Payment).where(
        Payment.merchant_id == merchant_id,
        Payment.external_id.in_(incoming_external_ids),
    )
    existing_records = {p.external_id: p for p in db.execute(existing_stmt).scalars().all()}

    # Track seen in current batch to handle intra-batch duplicate external_ids
    seen_in_batch: dict[str, IngestPaymentItem] = {}

    to_insert: list[Payment] = []

    for item in payload.payments:
        # Check intra-batch duplication
        if item.external_id in seen_in_batch:
            prev = seen_in_batch[item.external_id]
            if prev.amount == item.amount and prev.currency == item.currency:
                idempotent_skipped_count += 1
                results.append(
                    BatchIngestResultItem(
                        external_id=item.external_id,
                        status="IDEMPOTENT_SKIPPED",
                        payment_id=None,
                        detail="Duplicate within batch with identical payload",
                    )
                )
            else:
                conflict_count += 1
                results.append(
                    BatchIngestResultItem(
                        external_id=item.external_id,
                        status="PAYLOAD_CONFLICT",
                        payment_id=None,
                        detail=f"Intra-batch conflicting payload: amount {prev.amount} vs {item.amount}",
                    )
                )
            continue

        seen_in_batch[item.external_id] = item

        # Check existing in DB
        if item.external_id in existing_records:
            existing = existing_records[item.external_id]
            if existing.amount == item.amount and existing.currency == item.currency:
                idempotent_skipped_count += 1
                results.append(
                    BatchIngestResultItem(
                        external_id=item.external_id,
                        status="IDEMPOTENT_SKIPPED",
                        payment_id=existing.id,
                        detail="Existing payment record matched identically",
                    )
                )
            else:
                conflict_count += 1
                results.append(
                    BatchIngestResultItem(
                        external_id=item.external_id,
                        status="PAYLOAD_CONFLICT",
                        payment_id=existing.id,
                        detail=f"Payload conflict: existing {existing.amount} {existing.currency} vs incoming {item.amount} {item.currency}",
                    )
                )
        else:
            # Parse status
            try:
                status_val = PaymentStatus(item.status)
            except ValueError:
                status_val = PaymentStatus.CAPTURED

            new_id = uuid.uuid4()
            new_payment = Payment(
                id=new_id,
                merchant_id=merchant_id,
                external_id=item.external_id,
                amount=item.amount,
                currency=item.currency,
                status=status_val,
                payment_method=item.payment_method,
                captured_at=item.captured_at,
                source_created_at=item.source_created_at,
                payment_metadata=item.metadata,
            )
            to_insert.append(new_payment)
            inserted_count += 1
            results.append(
                BatchIngestResultItem(
                    external_id=item.external_id,
                    status="INSERTED",
                    payment_id=new_id,
                    detail="Successfully created new payment record",
                )
            )

    if to_insert:
        db.add_all(to_insert)

    if inserted_count > 0 or idempotent_skipped_count > 0 or conflict_count > 0:
        audit_entry = AuditEvent(
            id=uuid.uuid4(),
            merchant_id=merchant_id,
            actor_type=ActorType.HUMAN_CONTROLLER if actor_id else ActorType.SYSTEM,
            actor_id=actor_id,
            event_type="BATCH_PAYMENT_INGESTION",
            event_metadata={
                "total_received": len(payload.payments),
                "inserted": inserted_count,
                "idempotent_skipped": idempotent_skipped_count,
                "conflicts": conflict_count,
            },
        )
        db.add(audit_entry)

    db.commit()

    return BatchIngestResponse(
        total_received=len(payload.payments),
        inserted_count=inserted_count,
        idempotent_skipped_count=idempotent_skipped_count,
        conflict_count=conflict_count,
        results=results,
    )
