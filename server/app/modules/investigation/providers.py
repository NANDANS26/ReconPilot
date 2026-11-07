"""
Investigation Providers for ReconPilot Root-Cause Investigation.
Includes the Local ML Investigation Engine (RandomForestClassifier + feature extraction)
and a deterministic rule-grounded fallback provider.
All external cloud AI dependencies have been completely eliminated.
"""

import abc
import logging
from datetime import UTC, datetime

from app.core.config import Settings
from app.ml.feature_extractor import FeatureExtractor
from app.ml.inference import LocalMLInferenceService
from app.ml.model_registry import get_model_registry
from app.models.enums import CauseAssessment, EvidenceCategory, InvestigationStatus
from app.modules.evidence.schemas import ExceptionEvidenceResponse
from app.modules.investigation.schemas import (
    InvestigationReportData,
    LikelyCause,
    RecommendedAction,
)

logger = logging.getLogger(__name__)


class InvestigationProviderError(Exception):
    """Base exception for investigation provider failures."""

    pass


class ProviderUnavailableError(InvestigationProviderError):
    """Raised when the configured provider is unavailable."""

    pass


class ProviderExecutionError(InvestigationProviderError):
    """Raised when the provider fails to generate or return a valid response."""

    pass


class BaseInvestigationProvider(abc.ABC):
    """Abstract interface for investigation report generators."""

    @abc.abstractmethod
    async def investigate(
        self,
        dossier: ExceptionEvidenceResponse,
        notes: str | None = None,
    ) -> InvestigationReportData:
        """Analyzes the evidence dossier and returns a structured investigation report."""
        pass


class LocalMLInvestigationProvider(BaseInvestigationProvider):
    """
    Supervised Local Machine Learning Investigation Provider.
    Extracts structured financial features from the dossier, performs CPU inference
    via scikit-learn pipeline, and constructs grounded explanations using strictly verified facts.
    """

    def __init__(self, inference_service: LocalMLInferenceService | None = None) -> None:
        self.inference_service = inference_service or LocalMLInferenceService()

    async def investigate(
        self,
        dossier: ExceptionEvidenceResponse,
        notes: str | None = None,
    ) -> InvestigationReportData:
        recon = dossier.reconciliation
        summary = dossier.summary
        items = dossier.evidence_items

        # Index evidence by category for grounded citations
        category_map: dict[EvidenceCategory, list[str]] = {}
        for item in items:
            category_map.setdefault(item.category, []).append(item.evidence_id)

        payment_ids = category_map.get(EvidenceCategory.PAYMENT, [])
        settle_ids = category_map.get(EvidenceCategory.SETTLEMENT, [])
        fee_ids = category_map.get(EvidenceCategory.FEE, [])
        refund_ids = category_map.get(EvidenceCategory.REFUND, [])
        adj_ids = category_map.get(EvidenceCategory.ADJUSTMENT, [])
        recon_ids = category_map.get(EvidenceCategory.RECONCILIATION, [])
        conflict_ids = category_map.get(EvidenceCategory.RELATED_RECORD, [])

        # 1. Feature extraction
        feature_dict = FeatureExtractor.extract_from_dossier(dossier)

        # 2. Local ML Inference
        prediction = self.inference_service.predict(feature_dict)
        pred_cause = prediction.predicted_cause
        score = prediction.confidence_score

        supporting_evidence: list[str] = []
        contradicting_evidence: list[str] = []
        limitations: list[str] = []

        if prediction.inference_mode == "DETERMINISTIC_FALLBACK":
            limitations.append(
                f"[Degraded Mode] Local ML model unavailable ({prediction.abstention_reason}). "
                "Operating under deterministic fallback rules."
            )
        elif prediction.is_abstaining:
            limitations.append(f"[Model Abstention] {prediction.abstention_reason}")

        if notes:
            limitations.append(f"Operator note included: {notes}")

        # Formulate grounded explanations and actions based on predicted root cause
        likely_causes: list[LikelyCause] = []
        recommended_actions: list[RecommendedAction] = []

        # Verified financial facts
        exp_str = f"{recon.currency} {recon.expected_amount:,.2f}"
        act_str = f"{recon.currency} {recon.actual_amount:,.2f}"
        var_str = f"{recon.currency} {recon.variance:,.2f}"

        if pred_cause == "FEE_MISMATCH":
            cited = list(set(fee_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Discrepancy identified between configured merchant fee schedule and payment gateway deduction. "
                        f"Expected net: {exp_str}, Observed settlement: {act_str} (Variance: {var_str}). "
                        f"Local ML classifier identifies fee variance as the primary root cause with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.SUPPORTED
                    if score >= 0.80
                    else CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Verify payment gateway fee schedule agreement and contracted MDR rates.",
                    reason="Gateway fee deductions exceed expected calculated contract fees.",
                    supporting_evidence=list(set(fee_ids)),
                    requires_human_approval=True,
                )
            )

        elif pred_cause == "REFUND_MISMATCH":
            cited = list(set(refund_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Settlement refund deduction does not agree with recorded refund transactions. "
                        f"Expected net: {exp_str}, Observed: {act_str}. "
                        f"Local ML classifier identifies refund deduction discrepancy with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.SUPPORTED
                    if score >= 0.80
                    else CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Cross-reference customer refund ledger status with payment gateway refund API.",
                    reason="Variance between claimed refund and settled deduction amount.",
                    supporting_evidence=list(set(refund_ids)),
                    requires_human_approval=True,
                )
            )

        elif pred_cause == "ADJUSTMENT_MISMATCH":
            cited = list(set(adj_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Dispute, penalty, or fee adjustment in settlement statement differs from local accounting adjustments. "
                        f"Variance: {var_str}. Local ML classifier confidence: {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.SUPPORTED
                    if score >= 0.80
                    else CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Audit disputed chargeback or manual adjustment entries with payment provider.",
                    reason="Adjustment amounts differ between local accounting and settlement statement.",
                    supporting_evidence=list(set(adj_ids)),
                    requires_human_approval=True,
                )
            )

        elif pred_cause == "MISSING_PAYMENT":
            cited = list(set(settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Payment gateway credited settlement funds ({act_str}) without a corresponding internal payment capture. "
                        f"Local ML classifier ranks missing payment ledger entry with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Inspect gateway transaction logs for unrecorded payment callback or webhook drop.",
                    reason="Orphan settlement line item received without local ledger payment capture.",
                    supporting_evidence=cited,
                    requires_human_approval=True,
                )
            )

        elif pred_cause == "MISSING_SETTLEMENT":
            cited = list(set(payment_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Captured internal payment ({exp_str}) has not been deposited in any settlement batch. "
                        f"Local ML classifier ranks settlement missing/hold with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Check gateway settlement cycle (T+2/T+3) or pending hold status for this capture.",
                    reason="Payment captured internally but missing from gateway payout batches.",
                    supporting_evidence=cited,
                    requires_human_approval=True,
                )
            )

        elif pred_cause in ("DUPLICATE_PAYMENT", "DUPLICATE_SETTLEMENT"):
            cited = list(set(conflict_ids + payment_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Multiple financial records share the same external reference or transaction identifier. "
                        f"Local ML classifier identified duplicate record anomaly with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.SUPPORTED,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Investigate batch ingestion logs to determine whether duplicate webhook or CSV was ingested.",
                    reason="Conflicting records detected for same external transaction identifier.",
                    supporting_evidence=list(set(conflict_ids)),
                    requires_human_approval=True,
                )
            )

        elif pred_cause == "DELAYED_SETTLEMENT":
            cited = list(set(payment_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Settlement timing exceeded expected SLA window. "
                        f"Local ML classifier evaluated delay with confidence {score:.2f}."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Review gateway holiday schedule and banking transfer delays.",
                    reason="Settlement deposit lagged payment capture beyond standard processing window.",
                    supporting_evidence=cited,
                    requires_human_approval=True,
                )
            )

        else:
            # AMBIGUOUS_MATCH, INSUFFICIENT_EVIDENCE, AMOUNT_MISMATCH, or UNKNOWN
            cited = list(set(recon_ids + payment_ids + settle_ids + adj_ids + fee_ids + refund_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=(
                        f"Net variance ({var_str}) between expected ({exp_str}) and observed ({act_str}) amounts. "
                        f"Local ML investigation determined classification: '{pred_cause}' (score: {score:.2f}). "
                        "Evidence is ambiguous or requires manual financial controller review."
                    ),
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.UNDETERMINED,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Perform manual operator review of invoice, fee schedule, and gateway settlement file.",
                    reason="Discrepancy root cause could not be uniquely confirmed from ledger records alone.",
                    supporting_evidence=list(set(recon_ids)),
                    requires_human_approval=True,
                )
            )

        # Include ranked alternatives in limitations for operator transparency
        if prediction.ranked_alternatives:
            alt_summary = ", ".join(
                f"{alt['cause']} ({alt['probability']:.2f})"
                for alt in prediction.ranked_alternatives[1:3]
            )
            if alt_summary:
                limitations.append(f"Ranked alternative hypotheses: {alt_summary}")

        return InvestigationReportData(
            exception_id=dossier.exception_id,
            investigation_status=InvestigationStatus.COMPLETED,
            observed_discrepancy=summary.observed_discrepancy,
            likely_causes=likely_causes,
            supporting_evidence=sorted(list(set(supporting_evidence))),
            contradicting_evidence=sorted(list(set(contradicting_evidence))),
            missing_information=list(summary.missing_information),
            recommended_actions=recommended_actions,
            requires_human_review=True,
            limitations=limitations,
            generated_at=datetime.now(UTC),
        )


class MockInvestigationProvider(BaseInvestigationProvider):
    """
    Deterministic rule-grounded mock analyzer.
    Used for testing or when local ML artifact is uninitialized.
    """

    async def investigate(
        self,
        dossier: ExceptionEvidenceResponse,
        notes: str | None = None,
    ) -> InvestigationReportData:
        exc_type = dossier.reconciliation.exception_type
        items = dossier.evidence_items

        category_map: dict[EvidenceCategory, list[str]] = {}
        for item in items:
            category_map.setdefault(item.category, []).append(item.evidence_id)

        payment_ids = category_map.get(EvidenceCategory.PAYMENT, [])
        settle_ids = category_map.get(EvidenceCategory.SETTLEMENT, [])
        fee_ids = category_map.get(EvidenceCategory.FEE, [])
        recon_ids = category_map.get(EvidenceCategory.RECONCILIATION, [])

        likely_causes: list[LikelyCause] = []
        recommended_actions: list[RecommendedAction] = []
        supporting_evidence: list[str] = []
        contradicting_evidence: list[str] = []
        limitations: list[str] = ["Operating in deterministic fallback mode."]

        if notes:
            limitations.append(f"Operator note included: {notes}")

        recon = dossier.reconciliation
        exp_str = f"{recon.currency} {recon.expected_amount:,.2f}"
        act_str = f"{recon.currency} {recon.actual_amount:,.2f}"

        if exc_type.value == "FEE_MISMATCH":
            cited = list(set(fee_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=f"Discrepancy identified between merchant fee schedule and gateway settlement deduction (Expected: {exp_str}, Observed: {act_str}).",
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.SUPPORTED,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Verify payment gateway fee schedule agreement and MDR rate contract.",
                    reason="Gateway fee deductions exceed expected calculated fee.",
                    supporting_evidence=list(set(fee_ids)),
                    requires_human_approval=True,
                )
            )
        elif exc_type.value == "MISSING_PAYMENT":
            cited = list(set(settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=f"Gateway credited settlement funds ({act_str}) without corresponding internal payment record.",
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Inspect gateway transaction logs for unrecorded payment callback or webhook failure.",
                    reason="Orphan settlement item received without local ledger entry.",
                    supporting_evidence=cited,
                    requires_human_approval=True,
                )
            )
        elif exc_type.value == "MISSING_SETTLEMENT":
            cited = list(set(payment_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=f"Captured payment ({exp_str}) has not been deposited in any gateway settlement batch.",
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.PLAUSIBLE,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Check gateway settlement cycle (T+2/T+3) or pending hold status.",
                    reason="Payment captured internally but missing from settlement batches.",
                    supporting_evidence=cited,
                    requires_human_approval=True,
                )
            )
        else:
            cited = list(set(recon_ids + payment_ids + settle_ids))
            supporting_evidence.extend(cited)
            likely_causes.append(
                LikelyCause(
                    explanation=f"Net variance between expected ({exp_str}) and observed ({act_str}) amounts.",
                    supporting_evidence=cited,
                    contradicting_evidence=[],
                    assessment=CauseAssessment.UNDETERMINED,
                )
            )
            recommended_actions.append(
                RecommendedAction(
                    action="Perform manual operator review of invoice, fee schedule, and gateway settlement file.",
                    reason="Discrepancy root cause could not be uniquely confirmed from ledger records alone.",
                    supporting_evidence=list(set(recon_ids)),
                    requires_human_approval=True,
                )
            )

        return InvestigationReportData(
            exception_id=dossier.exception_id,
            investigation_status=InvestigationStatus.COMPLETED,
            observed_discrepancy=dossier.summary.observed_discrepancy,
            likely_causes=likely_causes,
            supporting_evidence=sorted(list(set(supporting_evidence))),
            contradicting_evidence=sorted(list(set(contradicting_evidence))),
            missing_information=list(dossier.summary.missing_information),
            recommended_actions=recommended_actions,
            requires_human_review=True,
            limitations=limitations,
            generated_at=datetime.now(UTC),
        )


def get_investigation_provider(settings: Settings) -> BaseInvestigationProvider:
    """
    Returns the configured investigation provider.
    Defaults to LocalMLInvestigationProvider.
    Falls back to MockInvestigationProvider if explicitly configured as 'mock'.
    Raises ProviderUnavailableError if an unconfigured or external provider is requested.
    """
    provider_type = (settings.AI_PROVIDER or "local_ml").lower()

    if provider_type in ("gemini", "openai", "claude", "ollama", "groq"):
        raise ProviderUnavailableError(
            f"Provider '{provider_type}' is not configured. External cloud LLMs have been removed; use 'local_ml'."
        )

    if provider_type == "mock":
        return MockInvestigationProvider()

    if provider_type in ("local_ml", "local", "ml"):
        registry = get_model_registry()
        if registry.is_degraded:
            logger.warning(
                f"Local ML model is in degraded mode: {registry.load_error}. "
                "Falling back to rule-grounded mock provider."
            )
            return MockInvestigationProvider()
        return LocalMLInvestigationProvider()

    raise ProviderUnavailableError(f"Provider '{provider_type}' is not supported.")
