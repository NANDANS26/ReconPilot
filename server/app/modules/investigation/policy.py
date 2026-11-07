"""
Deterministic Policy Gate for ReconPilot AI Investigation Engine.
Enforces non-negotiable safety constraints outside and independent of LLM reasoning.
"""

from decimal import Decimal

from app.models.enums import CauseAssessment
from app.modules.evidence.schemas import ExceptionEvidenceResponse
from app.modules.investigation.schemas import InvestigationReportData


class DeterministicPolicyGate:
    """
    Enforces boundary safety, ensures no financial mutations occur,
    and mandates human review for high-impact or ambiguous exceptions.
    """

    POLICY_NON_MUTATION = "POLICY_NON_MUTATION_ENFORCED"
    POLICY_NON_MUTATION_ENFORCED = "POLICY_NON_MUTATION_ENFORCED"
    POLICY_HUMAN_REVIEW_REQUIRED = "MANDATORY_HUMAN_REVIEW"
    POLICY_HIGH_FINANCIAL_IMPACT = "HIGH_FINANCIAL_IMPACT"
    POLICY_AMBIGUOUS_CAUSE = "AMBIGUOUS_OR_UNRESOLVED_CAUSE"
    POLICY_CONFLICTING_EVIDENCE = "CONFLICTING_RECORDS_DETECTED"
    POLICY_UNGROUNDED_EVIDENCE = "UNGROUNDED_CITATIONS_PRUNED"
    POLICY_MISSING_EVIDENCE = "INCOMPLETE_EVIDENCE_DOSSIER"

    @classmethod
    def evaluate(
        cls,
        report_data: InvestigationReportData,
        dossier: ExceptionEvidenceResponse,
        pruned_ids: list[str],
    ) -> tuple[InvestigationReportData, list[str]]:
        """
        Applies deterministic policy checks and returns (enforced_report, policy_flags).
        """
        policy_flags: list[str] = [cls.POLICY_NON_MUTATION]

        # 1. Non-mutation invariant: all recommended actions are strictly advisory
        for action in report_data.recommended_actions:
            action.requires_human_approval = True
            # If an action has no valid supporting evidence, explicitly flag it in the reason
            if not action.supporting_evidence:
                action.reason = f"[Policy: Uncorroborated Action] {action.reason}"

        # 2. Mandatory human review evaluation
        requires_human = False

        # Any monetary variance or impact > 0 triggers human review
        variance = abs(dossier.reconciliation.variance)
        if variance > Decimal("0.00"):
            requires_human = True
            policy_flags.append(cls.POLICY_HUMAN_REVIEW_REQUIRED)

        # High financial impact check: ALWAYS forces mandatory human review
        if abs(dossier.summary.expected_amount) > Decimal("1000.00") or variance > Decimal(
            "100.00"
        ):
            requires_human = True
            policy_flags.append(cls.POLICY_HIGH_FINANCIAL_IMPACT)

        # Ambiguous causes (PLAUSIBLE or UNDETERMINED)
        has_ambiguous_cause = any(
            cause.assessment in (CauseAssessment.PLAUSIBLE, CauseAssessment.UNDETERMINED)
            for cause in report_data.likely_causes
        )
        if has_ambiguous_cause or not report_data.likely_causes:
            requires_human = True
            policy_flags.append(cls.POLICY_AMBIGUOUS_CAUSE)

        # Conflicting records in dossier
        if dossier.summary.conflicts:
            requires_human = True
            policy_flags.append(cls.POLICY_CONFLICTING_EVIDENCE)

        # Missing information in dossier
        if dossier.summary.missing_information:
            requires_human = True
            policy_flags.append(cls.POLICY_MISSING_EVIDENCE)

        # Pruned hallucinated citations
        if pruned_ids:
            requires_human = True
            policy_flags.append(cls.POLICY_UNGROUNDED_EVIDENCE)

        # Force report flag deterministically
        report_data.requires_human_review = requires_human

        # Ensure observed discrepancy strictly reflects authoritative reconciliation
        if not report_data.observed_discrepancy:
            report_data.observed_discrepancy = dossier.summary.observed_discrepancy

        return report_data, sorted(list(set(policy_flags)))
