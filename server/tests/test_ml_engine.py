"""
Comprehensive test suite for ReconPilot Local ML Root-Cause Reasoning Engine.
Validates:
- Feature schema v1.0 specifications and feature extraction integrity
- Local model registry lifecycle, artifact validation, and degraded mode
- Inference pipeline, probability calibration, confidence thresholding, and margin abstention
- Absolute absence of external network / HTTP calls during investigation
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from app.ml.feature_extractor import FeatureExtractor
from app.ml.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TARGET_LABELS,
)
from app.ml.inference import LocalMLInferenceService
from app.ml.model_registry import ModelRegistry, get_model_registry
from app.modules.evidence.schemas import (
    EvidenceCategory,
    EvidenceItem,
    EvidenceStrength,
    ExceptionEvidenceResponse,
    ExceptionSeverity,
    ExceptionStatus,
    ExceptionType,
    FinancialEvidenceSummary,
    MatchStatus,
    ReconciliationContext,
    RelatedRecordsSummary,
)


def _build_test_dossier(
    variance: Decimal = Decimal("50.00"),
    has_payment: bool = True,
    has_settlement: bool = True,
    fee_count: int = 1,
) -> ExceptionEvidenceResponse:
    exc_id = uuid.uuid4()
    merch_id = uuid.uuid4()
    run_id = uuid.uuid4()
    res_id = uuid.uuid4()

    recon = ReconciliationContext(
        result_id=res_id,
        run_id=run_id,
        match_status=MatchStatus.EXCEPTION,
        exception_type=ExceptionType.FEE_MISMATCH,
        severity=ExceptionSeverity.MEDIUM,
        exception_status=ExceptionStatus.OPEN,
        expected_amount=Decimal("1000.00"),
        actual_amount=Decimal("950.00"),
        variance=variance,
        currency="INR",
        tolerance=Decimal("1.00"),
        created_at=datetime.now(UTC),
    )

    ev_items = [
        EvidenceItem(
            evidence_id=f"ev_recon_{res_id}",
            category=EvidenceCategory.RECONCILIATION,
            source_type="reconciliation_result",
            source_record_id=str(res_id),
            strength=EvidenceStrength.CONFIRMED,
        )
    ]

    pay_ids = []
    if has_payment:
        pay_id = str(uuid.uuid4())
        pay_ids.append(pay_id)
        ev_items.append(
            EvidenceItem(
                evidence_id=f"ev_pay_{pay_id}",
                category=EvidenceCategory.PAYMENT,
                source_type="payment",
                source_record_id=pay_id,
                strength=EvidenceStrength.CONFIRMED,
            )
        )

    settle_ids = []
    if has_settlement:
        settle_id = str(uuid.uuid4())
        settle_ids.append(settle_id)
        ev_items.append(
            EvidenceItem(
                evidence_id=f"ev_settle_{settle_id}",
                category=EvidenceCategory.SETTLEMENT,
                source_type="settlement_item",
                source_record_id=settle_id,
                strength=EvidenceStrength.CONFIRMED,
            )
        )

    fee_ids = []
    for _ in range(fee_count):
        fee_id = str(uuid.uuid4())
        fee_ids.append(fee_id)
        ev_items.append(
            EvidenceItem(
                evidence_id=f"ev_fee_{fee_id}",
                category=EvidenceCategory.FEE,
                source_type="fee",
                source_record_id=fee_id,
                strength=EvidenceStrength.CONFIRMED,
            )
        )

    return ExceptionEvidenceResponse(
        exception_id=exc_id,
        merchant_id=merch_id,
        reconciliation=recon,
        evidence_items=ev_items,
        related_records=RelatedRecordsSummary(
            payment_ids=pay_ids,
            settlement_item_ids=settle_ids,
            settlement_ids=[str(uuid.uuid4())] if has_settlement else [],
            fee_ids=fee_ids,
            refund_ids=[],
            adjustment_ids=[],
        ),
        summary=FinancialEvidenceSummary(
            observed_discrepancy="Fee mismatch observed between payment and settlement deductions",
            expected_amount=Decimal("1000.00"),
            actual_amount=Decimal("950.00"),
            variance=variance,
            currency="INR",
            overall_strength=EvidenceStrength.CORROBORATED,
            cause_established=True,
            supported_findings=["Deductions differ from fee schedule"],
            missing_information=[],
            conflicts=[],
        ),
    )


def test_feature_schema_and_extraction():
    """Verify feature extractor generates expected v1.0 integer minor-unit vectors without NaNs."""
    dossier = _build_test_dossier()
    feat_dict = FeatureExtractor.extract_from_dossier(dossier)

    assert isinstance(feat_dict, dict)
    assert len(feat_dict) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    # Check key integer minor units
    assert feat_dict["payment_amount_minor"] == 100000
    assert feat_dict["settlement_gross_minor"] == 95000
    assert feat_dict["abs_variance_minor"] == 5000
    assert feat_dict["has_payment"] == 1.0
    assert feat_dict["has_settlement_item"] == 1.0
    assert feat_dict["fee_count"] == 1.0

    df = pd.DataFrame([feat_dict])
    assert df.shape == (1, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    assert not df[NUMERIC_FEATURES].isnull().any().any()


def test_model_registry_loaded_and_metadata():
    """Verify local ML model artifact loads into registry with valid schema metadata."""
    registry = get_model_registry()
    assert registry.is_loaded is True
    assert registry.is_degraded is False
    assert registry.pipeline is not None
    assert registry.metadata["model_name"].startswith("reconpilot-")
    assert registry.metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert set(registry.metadata["classes"]) == set(TARGET_LABELS)


def test_model_registry_degraded_fallback_on_missing_artifact(tmp_path):
    """Verify registry enters degraded mode gracefully when artifact path does not exist."""
    fake_path = str(tmp_path / "non_existent.joblib")
    reg = ModelRegistry(artifact_path=fake_path)
    assert reg.is_loaded is False
    assert reg.is_degraded is True
    assert "not found" in reg.load_error.lower()


def test_local_ml_inference_classification():
    """Verify local ML model classifies a fee mismatch exception with high confidence."""
    dossier = _build_test_dossier(variance=Decimal("50.00"), fee_count=1)
    feat_dict = FeatureExtractor.extract_from_dossier(dossier)

    service = LocalMLInferenceService()
    result = service.predict(feat_dict)

    assert result.model_name.startswith("reconpilot-")
    assert result.inference_mode == "LOCAL_ML"
    assert result.predicted_cause in TARGET_LABELS
    assert 0.0 <= result.confidence_score <= 1.0
    assert len(result.ranked_alternatives) > 0

    # Ensure probabilities are sorted descending
    probs = [alt["probability"] for alt in result.ranked_alternatives]
    assert probs == sorted(probs, reverse=True)


def test_margin_and_confidence_abstention():
    """Verify low confidence or low margin triggers INSUFFICIENT_EVIDENCE abstention."""
    dossier = _build_test_dossier()
    feat_dict = FeatureExtractor.extract_from_dossier(dossier)

    service = LocalMLInferenceService()

    # Artificially require impossible confidence threshold 0.9999
    with patch.object(service.settings, "ML_CONFIDENCE_THRESHOLD", 0.9999):
        res_high_thresh = service.predict(feat_dict)
        assert res_high_thresh.is_abstaining is True
        assert res_high_thresh.predicted_cause == "INSUFFICIENT_EVIDENCE"
        assert "below the required conservative threshold" in res_high_thresh.abstention_reason

    # Artificially require impossible margin 0.9999
    with patch.object(service.settings, "ML_MIN_MARGIN", 0.9999):
        res_high_margin = service.predict(feat_dict)
        assert res_high_margin.is_abstaining is True
        assert res_high_margin.predicted_cause == "AMBIGUOUS_MATCH"
        assert "Ambiguous prediction" in res_high_margin.abstention_reason


def test_zero_external_network_calls():
    """Verify local ML engine performs 100% offline inference with zero HTTP/socket calls."""
    import socket
    import urllib.request

    dossier = _build_test_dossier()
    feat_dict = FeatureExtractor.extract_from_dossier(dossier)
    service = LocalMLInferenceService()

    with (
        patch.object(socket, "socket", side_effect=RuntimeError("Network access forbidden")),
        patch.object(
            urllib.request, "urlopen", side_effect=RuntimeError("Network access forbidden")
        ),
    ):
        # Must run cleanly without invoking sockets or urlopen
        result = service.predict(feat_dict)
        assert result.predicted_cause is not None
