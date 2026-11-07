"""
Comprehensive test suite for ReconPilot Immutable Evidence Snapshots.
Validates:
- Canonical JSON serialization of authoritative financial dossiers
- Deterministic SHA-256 integrity hash generation
- Tamper sensitivity: any mutation produces a differing hash
- Evidence citation verification in investigations
- Investigation report persistence of snapshot and hash in PostgreSQL
"""

import copy
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.db.session import SessionLocal
from app.models.enums import (
    EvidenceCategory,
    EvidenceStrength,
    ExceptionSeverity,
    ExceptionStatus,
    ExceptionType,
    InvestigationStatus,
    MatchStatus,
)
from app.models.exception import ExceptionRecord
from app.models.investigation_report import InvestigationReport
from app.modules.evidence.schemas import (
    EvidenceItem,
    ExceptionEvidenceResponse,
    FinancialEvidenceSummary,
    ReconciliationContext,
    RelatedRecordsSummary,
)
from app.modules.evidence.snapshot import canonicalize_snapshot, compute_snapshot_hash
from app.modules.investigation.schemas import StartInvestigationRequest
from app.modules.investigation.service import InvestigationService

NOW = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)


def _build_sample_dossier() -> ExceptionEvidenceResponse:
    exc_id = uuid.uuid4()
    merch_id = uuid.uuid4()
    res_id = uuid.uuid4()
    pay_id = uuid.uuid4()

    recon = ReconciliationContext(
        result_id=res_id,
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
        created_at=NOW,
    )

    ev_items = [
        EvidenceItem(
            evidence_id="ev_recon_test_1",
            category=EvidenceCategory.RECONCILIATION,
            source_type="reconciliation_result",
            source_record_id=str(res_id),
            strength=EvidenceStrength.CONFIRMED,
        ),
        EvidenceItem(
            evidence_id="ev_pay_test_2",
            category=EvidenceCategory.PAYMENT,
            source_type="payment",
            source_record_id=str(pay_id),
            strength=EvidenceStrength.CONFIRMED,
        ),
    ]

    return ExceptionEvidenceResponse(
        exception_id=exc_id,
        merchant_id=merch_id,
        reconciliation=recon,
        evidence_items=ev_items,
        related_records=RelatedRecordsSummary(
            payment_ids=[str(pay_id)],
            settlement_item_ids=[],
            settlement_ids=[],
            fee_ids=[],
            refund_ids=[],
            adjustment_ids=[],
        ),
        summary=FinancialEvidenceSummary(
            observed_discrepancy="Discrepancy of 100.00 INR",
            expected_amount=Decimal("1500.00"),
            actual_amount=Decimal("1400.00"),
            variance=Decimal("100.00"),
            currency="INR",
            overall_strength=EvidenceStrength.CONFIRMED,
            cause_established=True,
            supported_findings=["Under-settled amount"],
            missing_information=[],
            conflicts=[],
        ),
    )


def test_canonical_snapshot_determinism():
    """Serializing the same dossier twice produces identical dictionary and identical SHA-256 hash."""
    dossier = _build_sample_dossier()

    snap1 = canonicalize_snapshot(dossier, captured_at=NOW)
    snap2 = canonicalize_snapshot(dossier, captured_at=NOW)

    assert snap1 == snap2
    hash1 = compute_snapshot_hash(snap1)
    hash2 = compute_snapshot_hash(snap2)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string


def test_tamper_sensitivity():
    """Any alteration to the snapshot payload causes the hash to diverge completely."""
    dossier = _build_sample_dossier()
    original_snap = canonicalize_snapshot(dossier, captured_at=NOW)
    original_hash = compute_snapshot_hash(original_snap)

    # Tamper with expected amount by 1 cent
    tampered_snap = copy.deepcopy(original_snap)
    tampered_snap["reconciliation"]["expected_amount"] = "1500.01"

    tampered_hash = compute_snapshot_hash(tampered_snap)
    assert tampered_hash != original_hash


@pytest.mark.anyio
async def test_investigation_persists_evidence_snapshot():
    """Verify end-to-end investigation run saves snapshot and cryptographic hash to PostgreSQL."""
    db = SessionLocal()
    try:
        exc = db.query(ExceptionRecord).first()
        assert exc is not None

        service = InvestigationService(db=db, merchant_id=exc.merchant_id)
        result = await service.run_investigation(
            exception_id=exc.id,
            request=StartInvestigationRequest(force_refresh=True),
        )

        assert result.status == InvestigationStatus.COMPLETED
        assert result.evidence_snapshot_hash is not None
        assert len(result.evidence_snapshot_hash) == 64

        # Verify DB record has snapshot and verified matching hash
        report_db = (
            db.query(InvestigationReport).filter(InvestigationReport.id == result.id).first()
        )
        assert report_db is not None
        assert report_db.evidence_snapshot is not None
        assert report_db.evidence_snapshot_hash == result.evidence_snapshot_hash

        # Recomputing hash from DB snapshot matches saved hash
        recomputed = compute_snapshot_hash(report_db.evidence_snapshot)
        assert recomputed == report_db.evidence_snapshot_hash
    finally:
        db.close()
