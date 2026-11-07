"""
Core orchestration service for ReconPilot Local ML Investigation Engine.
Coordinates evidence retrieval, canonical snapshotting, local ML inference,
grounding validation, policy enforcement, and PostgreSQL persistence.
All external cloud AI dependencies have been eliminated.
"""

import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.feature_schema import FEATURE_SCHEMA_VERSION
from app.models.enums import InvestigationStatus
from app.models.exception import ExceptionRecord
from app.models.investigation_report import InvestigationReport
from app.modules.evidence.schemas import ExceptionEvidenceResponse
from app.modules.evidence.service import EvidenceCollector, EvidenceCorrelator
from app.modules.evidence.snapshot import canonicalize_snapshot, compute_snapshot_hash
from app.modules.investigation.grounding import EvidenceGroundingValidator
from app.modules.investigation.policy import DeterministicPolicyGate
from app.modules.investigation.providers import (
    InvestigationProviderError,
    LocalMLInvestigationProvider,
    ProviderExecutionError,
    ProviderUnavailableError,
    get_investigation_provider,
)
from app.modules.investigation.schemas import (
    InvestigationReportData,
    InvestigationResponse,
    StartInvestigationRequest,
)

logger = logging.getLogger(__name__)


class InvestigationService:
    """
    Coordinates Local ML investigations for exceptions under strict tenant isolation.
    """

    def __init__(self, db: Session, merchant_id: uuid.UUID):
        self.db = db
        self.merchant_id = merchant_id
        self.settings = get_settings()

    async def run_investigation(
        self,
        exception_id: uuid.UUID,
        request: StartInvestigationRequest | None = None,
    ) -> InvestigationResponse:
        """
        Executes an investigation for the specified exception.
        Adheres to non-mutation rules, immutable evidence snapshotting, and tenant isolation.
        """
        force_refresh = request.force_refresh if request else False
        notes = request.notes if request else None

        # 1. Enforce tenant isolation: verify exception exists for merchant
        exc = self.db.scalar(
            select(ExceptionRecord).where(
                ExceptionRecord.id == exception_id,
                ExceptionRecord.merchant_id == self.merchant_id,
            )
        )
        if not exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exception record {exception_id} not found.",
            )

        # 2. Check for recent existing completed investigation to avoid duplicate runs
        if not force_refresh:
            existing = self.db.scalar(
                select(InvestigationReport)
                .where(
                    InvestigationReport.exception_id == exception_id,
                    InvestigationReport.merchant_id == self.merchant_id,
                    InvestigationReport.status == InvestigationStatus.COMPLETED,
                )
                .order_by(InvestigationReport.created_at.desc())
            )
            if existing:
                return self._to_response(existing)

        # 3. Retrieve evidence dossier
        collector = EvidenceCollector(db=self.db, merchant_id=self.merchant_id)
        raw_evidence = collector.collect(exception_id=exception_id)
        if not raw_evidence:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Authoritative reconciliation result or evidence missing for exception {exception_id}.",
            )
        dossier = EvidenceCorrelator.correlate(raw_evidence)

        # 4. Generate immutable evidence snapshot with SHA-256 integrity hash
        snapshot = canonicalize_snapshot(dossier)
        snapshot_hash = compute_snapshot_hash(snapshot)

        # 5. Resolve provider
        provider_name = (self.settings.AI_PROVIDER or "local_ml").lower()
        model_name = self.settings.ML_MODEL_NAME
        prompt_version = "ml-v1.0"

        try:
            provider = get_investigation_provider(self.settings)
        except ProviderUnavailableError as err:
            logger.warning(
                f"Investigation Provider unavailable for exception {exception_id}: {err}"
            )
            return self._record_unavailable_investigation(
                exception_id=exception_id,
                dossier=dossier,
                snapshot=snapshot,
                snapshot_hash=snapshot_hash,
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
                error_msg=str(err),
            )

        # 6. Execute analysis
        try:
            report_data = await provider.investigate(dossier=dossier, notes=notes)
        except ProviderUnavailableError as err:
            logger.warning(f"Provider unavailable during call: {err}")
            return self._record_unavailable_investigation(
                exception_id=exception_id,
                dossier=dossier,
                snapshot=snapshot,
                snapshot_hash=snapshot_hash,
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
                error_msg=str(err),
            )
        except (ProviderExecutionError, InvestigationProviderError, Exception) as err:
            logger.error(f"Provider execution failure: {err}")
            return self._record_failed_investigation(
                exception_id=exception_id,
                dossier=dossier,
                snapshot=snapshot,
                snapshot_hash=snapshot_hash,
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
                error_msg=str(err),
            )

        # 7. Apply evidence grounding checks
        sanitized_report, pruned_ids, valid_cited_ids = (
            EvidenceGroundingValidator.validate_and_sanitize(report_data, dossier)
        )

        # 8. Apply deterministic policy gate
        enforced_report, policy_flags = DeterministicPolicyGate.evaluate(
            sanitized_report, dossier, pruned_ids
        )

        inference_mode = (
            "LOCAL_ML"
            if isinstance(provider, LocalMLInvestigationProvider)
            else "DETERMINISTIC_FALLBACK"
        )

        # 9. Persist report record with evidence snapshot in PostgreSQL
        record = InvestigationReport(
            merchant_id=self.merchant_id,
            exception_id=exception_id,
            status=InvestigationStatus.COMPLETED,
            provider=provider_name,
            model_name=model_name,
            prompt_version=prompt_version,
            observed_discrepancy=enforced_report.observed_discrepancy,
            report_data=enforced_report.model_dump(mode="json"),
            cited_evidence_ids=valid_cited_ids,
            requires_human_review=enforced_report.requires_human_review,
            policy_flags=policy_flags,
            evidence_snapshot=snapshot,
            evidence_snapshot_hash=snapshot_hash,
            model_version=model_name,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_score=Decimal("0.95") if inference_mode == "LOCAL_ML" else Decimal("0.75"),
            inference_mode=inference_mode,
            error_message=None,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        return self._to_response(record)

    def get_latest_investigation(
        self,
        exception_id: uuid.UUID,
    ) -> InvestigationResponse | None:
        """
        Retrieves the most recent investigation report for this exception.
        Enforces tenant isolation.
        """
        exc = self.db.scalar(
            select(ExceptionRecord).where(
                ExceptionRecord.id == exception_id,
                ExceptionRecord.merchant_id == self.merchant_id,
            )
        )
        if not exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exception record {exception_id} not found.",
            )

        record = self.db.scalar(
            select(InvestigationReport)
            .where(
                InvestigationReport.exception_id == exception_id,
                InvestigationReport.merchant_id == self.merchant_id,
            )
            .order_by(InvestigationReport.created_at.desc())
        )
        if not record:
            return None

        return self._to_response(record)

    def _record_unavailable_investigation(
        self,
        exception_id: uuid.UUID,
        dossier: ExceptionEvidenceResponse,
        snapshot: dict,
        snapshot_hash: str,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        error_msg: str,
    ) -> InvestigationResponse:
        empty_report = InvestigationReportData(
            exception_id=exception_id,
            investigation_status=InvestigationStatus.UNAVAILABLE,
            observed_discrepancy=dossier.summary.observed_discrepancy,
            likely_causes=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            missing_information=list(dossier.summary.missing_information),
            recommended_actions=[],
            requires_human_review=True,
            limitations=[f"Investigation provider unavailable: {error_msg}"],
            generated_at=datetime.now(UTC),
        )
        record = InvestigationReport(
            merchant_id=self.merchant_id,
            exception_id=exception_id,
            status=InvestigationStatus.UNAVAILABLE,
            provider=provider_name,
            model_name=model_name,
            prompt_version=prompt_version,
            observed_discrepancy=dossier.summary.observed_discrepancy,
            report_data=empty_report.model_dump(mode="json"),
            cited_evidence_ids=[],
            requires_human_review=True,
            policy_flags=[
                DeterministicPolicyGate.POLICY_NON_MUTATION,
                DeterministicPolicyGate.POLICY_HUMAN_REVIEW_REQUIRED,
            ],
            evidence_snapshot=snapshot,
            evidence_snapshot_hash=snapshot_hash,
            inference_mode="UNAVAILABLE",
            error_message=error_msg,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return self._to_response(record)

    def _record_failed_investigation(
        self,
        exception_id: uuid.UUID,
        dossier: ExceptionEvidenceResponse,
        snapshot: dict,
        snapshot_hash: str,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        error_msg: str,
    ) -> InvestigationResponse:
        failed_report = InvestigationReportData(
            exception_id=exception_id,
            investigation_status=InvestigationStatus.FAILED,
            observed_discrepancy=dossier.summary.observed_discrepancy,
            likely_causes=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            missing_information=list(dossier.summary.missing_information),
            recommended_actions=[],
            requires_human_review=True,
            limitations=[f"Investigation execution failed: {error_msg}"],
            generated_at=datetime.now(UTC),
        )
        record = InvestigationReport(
            merchant_id=self.merchant_id,
            exception_id=exception_id,
            status=InvestigationStatus.FAILED,
            provider=provider_name,
            model_name=model_name,
            prompt_version=prompt_version,
            observed_discrepancy=dossier.summary.observed_discrepancy,
            report_data=failed_report.model_dump(mode="json"),
            cited_evidence_ids=[],
            requires_human_review=True,
            policy_flags=[
                DeterministicPolicyGate.POLICY_NON_MUTATION,
                DeterministicPolicyGate.POLICY_HUMAN_REVIEW_REQUIRED,
            ],
            evidence_snapshot=snapshot,
            evidence_snapshot_hash=snapshot_hash,
            inference_mode="FAILED",
            error_message=error_msg,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return self._to_response(record)

    def _to_response(self, record: InvestigationReport) -> InvestigationResponse:
        report_data = InvestigationReportData.model_validate(record.report_data)
        return InvestigationResponse(
            id=record.id,
            exception_id=record.exception_id,
            merchant_id=record.merchant_id,
            status=InvestigationStatus(record.status),
            provider=record.provider,
            model_name=record.model_name,
            prompt_version=record.prompt_version,
            report=report_data,
            policy_flags=record.policy_flags or [],
            requires_human_review=record.requires_human_review,
            evidence_snapshot_hash=record.evidence_snapshot_hash,
            model_score=float(record.model_score) if record.model_score is not None else None,
            inference_mode=record.inference_mode,
            error_message=record.error_message,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
