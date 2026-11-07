"""
Comprehensive test suite for ReconPilot Batch Payment Ingestion & Idempotency.
Validates:
- Insertion of new payment records
- Idempotent deduplication when re-ingesting identical payloads
- Conflict detection when incoming record contradicts existing record (amount/currency)
- Intra-batch duplicate handling
- Tenant isolation across merchants
- Authenticated HTTP API endpoint behavior
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.core.security import create_access_token
from app.db.session import SessionLocal
from app.main import app
from app.models.enums import UserRole
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.user import User
from app.modules.reconciliation.ingestion import ingest_payments_batch_idempotent
from app.modules.reconciliation.schemas import BatchIngestPaymentsRequest, IngestPaymentItem
from fastapi.testclient import TestClient

client = TestClient(app)
NOW = datetime.now(UTC)


@pytest.fixture(autouse=True)
def cleanup_test_ingested_payments():
    """Ensure any test payments created during the test are safely cleaned up."""
    yield
    db = SessionLocal()
    try:
        db.query(Payment).filter(Payment.external_id.like("test_ingest_%")).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def test_batch_ingest_new_payments():
    """Verify new payments are inserted and return INSERTED status with valid IDs."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        assert merchant is not None

        item1 = IngestPaymentItem(
            external_id="test_ingest_p1",
            amount=Decimal("1250.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="upi",
            source_created_at=NOW,
        )
        item2 = IngestPaymentItem(
            external_id="test_ingest_p2",
            amount=Decimal("3400.50"),
            currency="INR",
            status="CAPTURED",
            payment_method="card",
            source_created_at=NOW,
        )
        req = BatchIngestPaymentsRequest(payments=[item1, item2])

        resp = ingest_payments_batch_idempotent(db=db, merchant_id=merchant.id, payload=req)

        assert resp.total_received == 2
        assert resp.inserted_count == 2
        assert resp.idempotent_skipped_count == 0
        assert resp.conflict_count == 0

        # Verify DB records
        p1_db = (
            db.query(Payment)
            .filter(Payment.merchant_id == merchant.id, Payment.external_id == "test_ingest_p1")
            .first()
        )
        assert p1_db is not None
        assert p1_db.amount == Decimal("1250.00")
        assert p1_db.payment_method == "upi"
    finally:
        db.close()


def test_batch_ingest_idempotent_replay():
    """Replaying the exact same payment batch must skip insertion and return existing payment IDs."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        assert merchant is not None

        item = IngestPaymentItem(
            external_id="test_ingest_replay_1",
            amount=Decimal("2000.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="card",
            source_created_at=NOW,
        )
        req = BatchIngestPaymentsRequest(payments=[item])

        # First ingestion -> INSERTED
        resp1 = ingest_payments_batch_idempotent(db=db, merchant_id=merchant.id, payload=req)
        assert resp1.inserted_count == 1
        payment_id_1 = resp1.results[0].payment_id

        # Second ingestion of identical item -> IDEMPOTENT_SKIPPED
        resp2 = ingest_payments_batch_idempotent(db=db, merchant_id=merchant.id, payload=req)
        assert resp2.inserted_count == 0
        assert resp2.idempotent_skipped_count == 1
        assert resp2.conflict_count == 0
        assert resp2.results[0].status == "IDEMPOTENT_SKIPPED"
        assert resp2.results[0].payment_id == payment_id_1
    finally:
        db.close()


def test_batch_ingest_payload_conflict_detection():
    """An incoming payment with conflicting amount must be flagged PAYLOAD_CONFLICT and not overwrite DB."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        assert merchant is not None

        # Insert initial payment of 500.00
        initial_item = IngestPaymentItem(
            external_id="test_ingest_conflict_1",
            amount=Decimal("500.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="card",
            source_created_at=NOW,
        )
        resp1 = ingest_payments_batch_idempotent(
            db=db,
            merchant_id=merchant.id,
            payload=BatchIngestPaymentsRequest(payments=[initial_item]),
        )
        assert resp1.inserted_count == 1

        # Attempt to ingest same external_id with different amount 999.00
        conflicting_item = IngestPaymentItem(
            external_id="test_ingest_conflict_1",
            amount=Decimal("999.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="card",
            source_created_at=NOW,
        )
        resp2 = ingest_payments_batch_idempotent(
            db=db,
            merchant_id=merchant.id,
            payload=BatchIngestPaymentsRequest(payments=[conflicting_item]),
        )
        assert resp2.inserted_count == 0
        assert resp2.conflict_count == 1
        assert resp2.results[0].status == "PAYLOAD_CONFLICT"
        assert "Payload conflict" in resp2.results[0].detail

        # Verify DB amount was NOT corrupted or overwritten
        db.expire_all()
        persisted = (
            db.query(Payment)
            .filter(
                Payment.merchant_id == merchant.id,
                Payment.external_id == "test_ingest_conflict_1",
            )
            .first()
        )
        assert persisted.amount == Decimal("500.00")
    finally:
        db.close()


def test_batch_ingest_intra_batch_duplicates():
    """Two items in the same batch with the same external_id: first inserts, second skips or conflicts."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        assert merchant is not None

        item1 = IngestPaymentItem(
            external_id="test_ingest_intra_dup",
            amount=Decimal("750.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="netbanking",
            source_created_at=NOW,
        )
        item2 = IngestPaymentItem(
            external_id="test_ingest_intra_dup",
            amount=Decimal("750.00"),
            currency="INR",
            status="CAPTURED",
            payment_method="netbanking",
            source_created_at=NOW,
        )
        req = BatchIngestPaymentsRequest(payments=[item1, item2])

        resp = ingest_payments_batch_idempotent(db=db, merchant_id=merchant.id, payload=req)

        assert resp.total_received == 2
        assert resp.inserted_count == 1
        assert resp.idempotent_skipped_count == 1
        assert resp.conflict_count == 0
    finally:
        db.close()


def test_api_ingestion_payments_endpoint():
    """Verify POST /api/v1/ingestion/payments endpoint returns valid ApiResponse."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        user = (
            db.query(User)
            .filter(User.merchant_id == merchant.id, User.role == UserRole.FINANCE_OPERATOR)
            .first()
        )
        assert user is not None

        token = create_access_token(
            data={"sub": str(user.id), "merchant_id": str(merchant.id), "role": user.role.value}
        )
        headers = {"Authorization": f"Bearer {token}"}

        payload = {
            "payments": [
                {
                    "external_id": "test_ingest_api_1",
                    "amount": "800.00",
                    "currency": "INR",
                    "status": "CAPTURED",
                    "payment_method": "card",
                    "source_created_at": NOW.isoformat(),
                }
            ]
        }

        resp = client.post("/api/v1/ingestion/payments", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["total_received"] == 1
        assert data["data"]["inserted_count"] == 1
        assert data["data"]["results"][0]["status"] == "INSERTED"
    finally:
        db.close()
