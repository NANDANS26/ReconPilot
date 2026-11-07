import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import CauseAssessment, InvestigationStatus


class LikelyCause(BaseModel):
    """A plausible or established root cause grounded in evidence."""

    model_config = ConfigDict(from_attributes=True)

    explanation: str = Field(description="Concise description of the suspected cause")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="IDs of evidence items that support this cause"
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list, description="IDs of evidence items that contradict this cause"
    )
    assessment: CauseAssessment = Field(
        description="Grounding level: SUPPORTED, PLAUSIBLE, or UNDETERMINED"
    )


class RecommendedAction(BaseModel):
    """A safe, advisory operator action grounded in findings."""

    model_config = ConfigDict(from_attributes=True)

    action: str = Field(description="Description of the suggested operational next step")
    reason: str = Field(description="Justification for this recommendation")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence references supporting this action"
    )
    requires_human_approval: bool = Field(
        default=True, description="Strictly true: all operational actions require human sign-off"
    )


class InvestigationReportData(BaseModel):
    """Structured, evidence-grounded investigation report."""

    model_config = ConfigDict(from_attributes=True)

    exception_id: uuid.UUID
    investigation_status: InvestigationStatus
    observed_discrepancy: str
    likely_causes: list[LikelyCause] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    requires_human_review: bool = Field(default=True)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InvestigationResponse(BaseModel):
    """Complete API response for an exception investigation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    exception_id: uuid.UUID
    merchant_id: uuid.UUID
    status: InvestigationStatus
    provider: str
    model_name: str
    prompt_version: str
    report: InvestigationReportData
    policy_flags: list[str] = Field(default_factory=list)
    requires_human_review: bool = True
    evidence_snapshot_hash: str | None = None
    model_score: float | None = None
    inference_mode: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class StartInvestigationRequest(BaseModel):
    """Optional parameters when triggering an investigation."""

    force_refresh: bool = Field(
        default=False, description="Whether to bypass cached recent investigation report"
    )
    notes: str | None = Field(
        default=None, description="Optional operator context (treated strictly as untrusted input)"
    )
