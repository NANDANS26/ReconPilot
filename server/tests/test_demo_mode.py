import uuid
from datetime import timedelta

from app.core.security import create_access_token
from app.db.session import SessionLocal
from app.main import app
from app.models.enums import UserRole
from app.models.exception import ExceptionRecord
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.settlement import Settlement
from app.models.user import User
from app.modules.demo.service import DemoService
from fastapi.testclient import TestClient

client = TestClient(app)


def test_demo_scenarios_derivation_from_real_records():
    """
    Verifies DemoService derives scenario references from actual accessible records,
    without fabrication or hardcoded identifiers.
    """
    db = SessionLocal()
    try:
        demo_merchant_id = uuid.UUID("00000000-0000-4000-a000-000000000001")
        service = DemoService(db=db, merchant_id=demo_merchant_id)
        response = service.get_scenarios()

        assert response.is_demo_mode is True
        assert response.merchant_id == demo_merchant_id
        assert len(response.scenarios) == 7  # 6 standard + 1 conflicting evidence

        # Check that available scenarios point to real records in this merchant
        available_scenarios = [s for s in response.scenarios if s.available]
        assert len(available_scenarios) > 0

        for s in available_scenarios:
            assert s.exception_id is not None
            # Verify the record exists in the database under this merchant
            exc = db.get(ExceptionRecord, s.exception_id)
            assert exc is not None
            assert exc.merchant_id == demo_merchant_id
            assert s.currency == exc.currency
    finally:
        db.close()


def test_demo_scenarios_empty_merchant_isolation():
    """
    Verifies that a merchant with zero exceptions gets available=False for all scenarios,
    proving no cross-tenant leaking or fake fallback records.
    """
    db = SessionLocal()
    try:
        isolated_merchant = Merchant(
            name="Empty Demo Isolated",
            email=f"empty_demo_{uuid.uuid4().hex[:6]}@test.local",
            currency="USD",
            timezone="UTC",
        )
        db.add(isolated_merchant)
        db.flush()

        service = DemoService(db=db, merchant_id=isolated_merchant.id)
        response = service.get_scenarios()

        assert response.merchant_id == isolated_merchant.id
        assert response.total_available == 0
        for s in response.scenarios:
            assert s.available is False
            assert s.exception_id is None
    finally:
        db.rollback()
        db.close()


def test_demo_scenarios_api_endpoint():
    """
    Tests the GET /api/v1/demo/scenarios endpoint with authenticated headers.
    """
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        user = (
            db.query(User)
            .filter(User.merchant_id == merchant.id, User.role == UserRole.REVIEWER)
            .first()
        )
        assert user is not None

        token = create_access_token(
            data={"sub": str(user.id), "merchant_id": str(merchant.id), "role": user.role.value},
            expires_delta=timedelta(minutes=30),
        )
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.get("/api/v1/demo/scenarios", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "data" in data
        res = data["data"]
        assert res["is_demo_mode"] is True
        assert "scenarios" in res
        assert len(res["scenarios"]) == 7
    finally:
        db.close()


def test_demo_no_financial_mutation_guarantee():
    """
    Guarantees querying demo scenarios never mutates financial balances or records.
    """
    db = SessionLocal()
    try:
        demo_merchant_id = uuid.UUID("00000000-0000-4000-a000-000000000001")
        service = DemoService(db=db, merchant_id=demo_merchant_id)

        # Count initial records
        pay_count_before = db.query(Payment).filter(Payment.merchant_id == demo_merchant_id).count()
        settle_count_before = (
            db.query(Settlement).filter(Settlement.merchant_id == demo_merchant_id).count()
        )
        exc_count_before = (
            db.query(ExceptionRecord)
            .filter(ExceptionRecord.merchant_id == demo_merchant_id)
            .count()
        )

        # Call demo service
        _ = service.get_scenarios()

        # Re-verify counts
        pay_count_after = db.query(Payment).filter(Payment.merchant_id == demo_merchant_id).count()
        settle_count_after = (
            db.query(Settlement).filter(Settlement.merchant_id == demo_merchant_id).count()
        )
        exc_count_after = (
            db.query(ExceptionRecord)
            .filter(ExceptionRecord.merchant_id == demo_merchant_id)
            .count()
        )

        assert pay_count_before == pay_count_after
        assert settle_count_before == settle_count_after
        assert exc_count_before == exc_count_after
    finally:
        db.close()
