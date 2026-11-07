"""
Comprehensive verification test suite for ReconPilot AI Investigation Engine (Phase 06).
Validates schema adherence, evidence grounding, policy gating, tenant isolation,
error handling, and non-mutation guarantees.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from app.core.config import Settings
from app.core.security import create_access_token
from app.db.session import SessionLocal
from app.main import app
from app.models.enums import (
    CauseAssessment,
    EvidenceCategory,
    EvidenceStrength,
    ExceptionSeverity,
    ExceptionStatus,
    ExceptionType,
    InvestigationStatus,
    MatchStatus,
    UserRole,
)
from app.models.exception import ExceptionRecord
from app.models.investigation_report import InvestigationReport
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.reconciliation_result import ReconciliationResult
from app.models.settlement import Settlement
from app.models.settlement_item import SettlementItem
from app.models.user import User
from app.modules.evidence.schemas import (
    EvidenceItem,
    ExceptionEvidenceResponse,
    FinancialEvidenceSummary,
    ReconciliationContext,
    RelatedRecordsSummary,
)
from app.modules.investigation.grounding import EvidenceGroundingValidator
from app.modules.investigation.policy import DeterministicPolicyGate
from app.modules.investigation.providers import (
    BaseInvestigationProvider,
    ProviderExecutionError,
)
from app.modules.investigation.schemas import (
    InvestigationReportData,
    LikelyCause,
    RecommendedAction,
    StartInvestigationRequest,
)
from app.modules.investigation.service import InvestigationService
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_investigation_test_records():
    """Ensures any test-created investigation records or users are cleaned up."""
    yield
    db = SessionLocal()
    try:
        db.query(InvestigationReport).filter(
            InvestigationReport.provider.in_(["mock", "test_mock", "failing_mock", "local_ml"])
        ).delete(synchronize_session=False)
        db.query(User).filter(User.email.like("test_inv_%")).delete(synchronize_session=False)
        db.query(Merchant).filter(Merchant.name.like("Test Inv Merchant%")).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@pytest.mark.anyio
async def test_valid_structured_investigation_flow():
    """Verify end-to-end investigation creation and persistence with mock provider."""
    db = SessionLocal()
    try:
        # Pick seeded exception for primary merchant
        exc = (
            db.query(ExceptionRecord)
            .filter(ExceptionRecord.exception_type == ExceptionType.FEE_MISMATCH)
            .first()
        )
        assert exc is not None

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        result = await service.run_investigation(
            exception_id=exc.id,
            request=StartInvestigationRequest(force_refresh=True),
        )

        assert result is not None
        assert result.status == InvestigationStatus.COMPLETED
        assert result.exception_id == exc.id
        assert result.merchant_id == exc.merchant_id
        assert len(result.report.likely_causes) > 0
        assert len(result.report.recommended_actions) > 0
        assert result.requires_human_review is True
        assert DeterministicPolicyGate.POLICY_NON_MUTATION_ENFORCED in result.policy_flags

        # Check DB record exists
        db_record = (
            db.query(InvestigationReport).filter(InvestigationReport.id == result.id).first()
        )
        assert db_record is not None
        assert db_record.status == InvestigationStatus.COMPLETED.value
    finally:
        db.close()


@pytest.mark.anyio
async def test_invalid_schema_output_handling():
    """Verify that malformed or schema-invalid LLM output is caught and marked FAILED."""
    db = SessionLocal()
    try:
        exc = db.query(ExceptionRecord).first()
        assert exc is not None

        class BrokenProvider(BaseInvestigationProvider):
            async def investigate(self, dossier, notes=None):
                raise ProviderExecutionError(
                    "Invalid JSON response: missing required field 'likely_causes'"
                )

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        with patch(
            "app.modules.investigation.service.get_investigation_provider",
            return_value=BrokenProvider(),
        ):
            result = await service.run_investigation(
                exception_id=exc.id,
                request=StartInvestigationRequest(force_refresh=True),
            )

        assert result.status == InvestigationStatus.FAILED
        assert result.error_message is not None
        assert "Invalid JSON response" in result.error_message
        assert result.report.investigation_status == InvestigationStatus.FAILED
    finally:
        db.close()


def test_unknown_or_fabricated_evidence_references_pruned():
    """Verify that hallucinated evidence IDs not in the dossier are pruned and flagged."""
    dossier = ExceptionEvidenceResponse(
        exception_id=uuid.uuid4(),
        merchant_id=uuid.uuid4(),
        reconciliation=ReconciliationContext(
            result_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.AMOUNT_MISMATCH,
            severity=ExceptionSeverity.HIGH,
            exception_status=ExceptionStatus.OPEN,
            expected_amount=Decimal("1500.00"),
            actual_amount=Decimal("1400.00"),
            variance=Decimal("100.00"),
            currency="INR",
            tolerance=Decimal("1.00"),
            created_at=datetime.now(UTC),
        ),
        evidence_items=[
            EvidenceItem(
                evidence_id="ev_recon_valid_123",
                category=EvidenceCategory.RECONCILIATION,
                source_type="reconciliation_result",
                source_record_id=str(uuid.uuid4()),
                strength=EvidenceStrength.CONFIRMED,
            ),
            EvidenceItem(
                evidence_id="ev_pay_valid_456",
                category=EvidenceCategory.PAYMENT,
                source_type="payment",
                source_record_id=str(uuid.uuid4()),
                strength=EvidenceStrength.CONFIRMED,
            ),
        ],
        related_records=RelatedRecordsSummary(),
        summary=FinancialEvidenceSummary(
            observed_discrepancy="Expected 1500.00 but got 1400.00",
            expected_amount=Decimal("1500.00"),
            actual_amount=Decimal("1400.00"),
            variance=Decimal("100.00"),
            currency="INR",
            overall_strength=EvidenceStrength.CORROBORATED,
            cause_established=True,
            supported_findings=["Ledger match"],
            missing_information=[],
            conflicts=[],
        ),
    )

    # Model fabricates two unknown IDs: 'hallucinated_tx_001' and 'phantom_ref_999'
    raw_report = InvestigationReportData(
        exception_id=dossier.exception_id,
        investigation_status=InvestigationStatus.COMPLETED,
        observed_discrepancy="Amount variance",
        likely_causes=[
            LikelyCause(
                explanation="Model fabricated an unknown transaction ref",
                supporting_evidence=["ev_recon_valid_123", "hallucinated_tx_001"],
                contradicting_evidence=["phantom_ref_999"],
                assessment=CauseAssessment.SUPPORTED,
            )
        ],
        supporting_evidence=["ev_pay_valid_456", "hallucinated_tx_001"],
        contradicting_evidence=[],
        missing_information=[],
        recommended_actions=[
            RecommendedAction(
                action="Verify hallucinated ref",
                reason="Model guessed it",
                supporting_evidence=["phantom_ref_999"],
                requires_human_approval=True,
            )
        ],
        requires_human_review=False,
        limitations=[],
        generated_at=datetime.now(UTC),
    )

    sanitized, pruned, valid_cited = EvidenceGroundingValidator.validate_and_sanitize(
        raw_report, dossier
    )

    # Verify hallucinated references were pruned
    assert "hallucinated_tx_001" in pruned
    assert "phantom_ref_999" in pruned
    assert "hallucinated_tx_001" not in sanitized.supporting_evidence
    assert "hallucinated_tx_001" not in sanitized.likely_causes[0].supporting_evidence
    assert "phantom_ref_999" not in sanitized.likely_causes[0].contradicting_evidence
    assert "phantom_ref_999" not in sanitized.recommended_actions[0].supporting_evidence

    # Valid IDs retained
    assert "ev_recon_valid_123" in sanitized.likely_causes[0].supporting_evidence
    assert "ev_pay_valid_456" in sanitized.supporting_evidence

    # Limitations alert appended
    assert any("[Evidence Policy Alert]" in lim for lim in sanitized.limitations)

    # Deterministic policy gate marks ungrounded citations pruned
    enforced, policy_flags = DeterministicPolicyGate.evaluate(sanitized, dossier, pruned)
    assert DeterministicPolicyGate.POLICY_UNGROUNDED_EVIDENCE in policy_flags
    assert enforced.requires_human_review is True


def test_conflicting_evidence_handling():
    """Verify that conflicting records in the evidence dossier escalate to human review."""
    dossier = ExceptionEvidenceResponse(
        exception_id=uuid.uuid4(),
        merchant_id=uuid.uuid4(),
        reconciliation=ReconciliationContext(
            result_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.DUPLICATE_PAYMENT,
            severity=ExceptionSeverity.HIGH,
            exception_status=ExceptionStatus.OPEN,
            expected_amount=Decimal("100.00"),
            actual_amount=Decimal("200.00"),
            variance=Decimal("100.00"),
            currency="INR",
            tolerance=Decimal("1.00"),
            created_at=datetime.now(UTC),
        ),
        evidence_items=[
            EvidenceItem(
                evidence_id="ev_conflict_1",
                category=EvidenceCategory.RELATED_RECORD,
                source_type="duplicate",
                strength=EvidenceStrength.CONFLICTING,
            )
        ],
        related_records=RelatedRecordsSummary(),
        summary=FinancialEvidenceSummary(
            observed_discrepancy="Duplicate payment detected",
            expected_amount=Decimal("100.00"),
            actual_amount=Decimal("200.00"),
            variance=Decimal("100.00"),
            currency="INR",
            overall_strength=EvidenceStrength.CONFLICTING,
            cause_established=True,
            supported_findings=[],
            missing_information=[],
            conflicts=["Duplicate gateway payments share the same transaction external ID"],
        ),
    )

    report = InvestigationReportData(
        exception_id=dossier.exception_id,
        investigation_status=InvestigationStatus.COMPLETED,
        observed_discrepancy=dossier.summary.observed_discrepancy,
        likely_causes=[
            LikelyCause(
                explanation="Duplicate payment",
                supporting_evidence=["ev_conflict_1"],
                contradicting_evidence=[],
                assessment=CauseAssessment.SUPPORTED,
            )
        ],
        supporting_evidence=["ev_conflict_1"],
        contradicting_evidence=[],
        missing_information=[],
        recommended_actions=[],
        requires_human_review=False,
        limitations=[],
        generated_at=datetime.now(UTC),
    )

    sanitized, pruned, _ = EvidenceGroundingValidator.validate_and_sanitize(report, dossier)
    enforced, policy_flags = DeterministicPolicyGate.evaluate(sanitized, dossier, pruned)

    assert DeterministicPolicyGate.POLICY_CONFLICTING_EVIDENCE in policy_flags
    assert enforced.requires_human_review is True
    assert any("[Deterministic Conflict]" in lim for lim in sanitized.limitations)


def test_undetermined_causes_and_missing_evidence():
    """Verify that undetermined causes and missing info trigger policy flags."""
    dossier = ExceptionEvidenceResponse(
        exception_id=uuid.uuid4(),
        merchant_id=uuid.uuid4(),
        reconciliation=ReconciliationContext(
            result_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            match_status=MatchStatus.EXCEPTION,
            exception_type=ExceptionType.MISSING_PAYMENT,
            severity=ExceptionSeverity.MEDIUM,
            exception_status=ExceptionStatus.OPEN,
            expected_amount=Decimal("0.00"),
            actual_amount=Decimal("500.00"),
            variance=Decimal("-500.00"),
            currency="INR",
            tolerance=Decimal("1.00"),
            created_at=datetime.now(UTC),
        ),
        evidence_items=[],
        related_records=RelatedRecordsSummary(),
        summary=FinancialEvidenceSummary(
            observed_discrepancy="Missing local payment record",
            expected_amount=Decimal("0.00"),
            actual_amount=Decimal("500.00"),
            variance=Decimal("-500.00"),
            currency="INR",
            overall_strength=EvidenceStrength.INCOMPLETE,
            cause_established=False,
            supported_findings=[],
            missing_information=["Internal payment ledger record is missing."],
            conflicts=[],
        ),
    )

    report = InvestigationReportData(
        exception_id=dossier.exception_id,
        investigation_status=InvestigationStatus.COMPLETED,
        observed_discrepancy=dossier.summary.observed_discrepancy,
        likely_causes=[
            LikelyCause(
                explanation="Undetermined orphan credit",
                supporting_evidence=[],
                contradicting_evidence=[],
                assessment=CauseAssessment.UNDETERMINED,
            )
        ],
        supporting_evidence=[],
        contradicting_evidence=[],
        missing_information=[],
        recommended_actions=[],
        requires_human_review=False,
        limitations=[],
        generated_at=datetime.now(UTC),
    )

    sanitized, pruned, _ = EvidenceGroundingValidator.validate_and_sanitize(report, dossier)
    enforced, policy_flags = DeterministicPolicyGate.evaluate(sanitized, dossier, pruned)

    assert DeterministicPolicyGate.POLICY_AMBIGUOUS_CAUSE in policy_flags
    assert DeterministicPolicyGate.POLICY_MISSING_EVIDENCE in policy_flags
    assert enforced.requires_human_review is True
    assert "Internal payment ledger record is missing." in sanitized.missing_information


@pytest.mark.anyio
async def test_provider_failure_and_timeout():
    """Verify provider timeout results in FAILED investigation status without crashing."""
    db = SessionLocal()
    try:
        exc = db.query(ExceptionRecord).first()
        assert exc is not None

        class TimeoutProvider(BaseInvestigationProvider):
            async def investigate(self, dossier, notes=None):
                raise ProviderExecutionError(
                    "Connection timeout while awaiting Gemini LLM response."
                )

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        with patch(
            "app.modules.investigation.service.get_investigation_provider",
            return_value=TimeoutProvider(),
        ):
            result = await service.run_investigation(
                exception_id=exc.id,
                request=StartInvestigationRequest(force_refresh=True),
            )

        assert result.status == InvestigationStatus.FAILED
        assert "timeout" in result.error_message.lower()
    finally:
        db.close()


@pytest.mark.anyio
async def test_provider_unavailable_when_unconfigured():
    """Verify provider returns UNAVAILABLE when credentials are not configured."""
    db = SessionLocal()
    try:
        exc = db.query(ExceptionRecord).first()
        assert exc is not None

        unconfigured_settings = Settings(
            AI_PROVIDER="gemini",
            LLM_API_KEY="",
        )

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        service.settings = unconfigured_settings

        result = await service.run_investigation(
            exception_id=exc.id,
            request=StartInvestigationRequest(force_refresh=True),
        )

        assert result.status == InvestigationStatus.UNAVAILABLE
        assert "not configured" in result.error_message
        assert result.report.investigation_status == InvestigationStatus.UNAVAILABLE
    finally:
        db.close()


def test_cross_tenant_access_denial():
    """Verify cross-tenant requests return 404 and cannot access other merchant investigations."""
    db = SessionLocal()
    try:
        merchant_a = db.query(Merchant).first()
        exc_a = (
            db.query(ExceptionRecord).filter(ExceptionRecord.merchant_id == merchant_a.id).first()
        )
        assert exc_a is not None

        # Create Merchant B
        merchant_b = Merchant(
            id=uuid.uuid4(),
            name=f"Test Inv Merchant B {uuid.uuid4().hex[:6]}",
            email=f"test_inv_b_{uuid.uuid4().hex[:6]}@example.com",
            currency="INR",
            timezone="Asia/Kolkata",
        )
        db.add(merchant_b)
        user_b = User(
            id=uuid.uuid4(),
            merchant_id=merchant_b.id,
            email=f"test_inv_user_{uuid.uuid4().hex[:6]}@example.com",
            password_hash="test_pw_hash",
            full_name="Tenant B Operator",
            role=UserRole.FINANCE_OPERATOR,
        )
        db.add(user_b)
        db.commit()

        token_b = create_access_token(
            data={
                "sub": str(user_b.id),
                "merchant_id": str(merchant_b.id),
                "role": user_b.role.value,
            },
            expires_delta=timedelta(minutes=30),
        )
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # User B attempts to start investigation for Merchant A's exception
        post_resp = client.post(
            f"/api/v1/exceptions/{exc_a.id}/investigations",
            json={"force_refresh": True},
            headers=headers_b,
        )
        assert post_resp.status_code == 404

        # User B attempts to get latest investigation for Merchant A's exception
        get_resp = client.get(
            f"/api/v1/exceptions/{exc_a.id}/investigations/latest",
            headers=headers_b,
        )
        assert get_resp.status_code == 404
    finally:
        db.close()


@pytest.mark.anyio
async def test_financial_records_non_mutation():
    """Verify that running investigations never mutates financial tables in PostgreSQL."""
    db = SessionLocal()
    try:
        exc = db.query(ExceptionRecord).first()
        assert exc is not None

        # Capture counts before
        payments_before = db.query(Payment).count()
        settlements_before = db.query(Settlement).count()
        items_before = db.query(SettlementItem).count()
        results_before = db.query(ReconciliationResult).count()
        exceptions_before = db.query(ExceptionRecord).count()

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        result = await service.run_investigation(
            exception_id=exc.id,
            request=StartInvestigationRequest(force_refresh=True),
        )
        assert result.status == InvestigationStatus.COMPLETED

        # Verify counts after
        assert db.query(Payment).count() == payments_before
        assert db.query(Settlement).count() == settlements_before
        assert db.query(SettlementItem).count() == items_before
        assert db.query(ReconciliationResult).count() == results_before
        assert db.query(ExceptionRecord).count() == exceptions_before

        # Verify all recommended actions strictly require human approval
        for action in result.report.recommended_actions:
            assert action.requires_human_approval is True
    finally:
        db.close()


def test_api_investigation_endpoints_e2e():
    """Verify HTTP POST and GET /investigations endpoints with authenticated user."""
    db = SessionLocal()
    try:
        merchant = db.query(Merchant).first()
        user = (
            db.query(User)
            .filter(User.merchant_id == merchant.id, User.role == UserRole.ADMIN)
            .first()
        )
        assert user is not None

        exc = db.query(ExceptionRecord).filter(ExceptionRecord.merchant_id == merchant.id).first()
        assert exc is not None

        token = create_access_token(
            data={
                "sub": str(user.id),
                "merchant_id": str(merchant.id),
                "role": user.role.value,
            },
            expires_delta=timedelta(minutes=30),
        )
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Trigger POST investigation
        post_resp = client.post(
            f"/api/v1/exceptions/{exc.id}/investigations",
            json={"force_refresh": True, "notes": "E2E API test investigation"},
            headers=headers,
        )
        assert post_resp.status_code == 200
        post_data = post_resp.json()
        assert post_data["success"] is True
        assert post_data["data"]["exception_id"] == str(exc.id)
        assert post_data["data"]["status"] == "COMPLETED"
        assert "likely_causes" in post_data["data"]["report"]

        # 2. Retrieve GET latest investigation
        get_resp = client.get(
            f"/api/v1/exceptions/{exc.id}/investigations/latest",
            headers=headers,
        )
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["success"] is True
        assert get_data["data"]["id"] == post_data["data"]["id"]
        assert get_data["data"]["requires_human_review"] is True
    finally:
        db.close()
