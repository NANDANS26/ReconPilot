import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.exception import ExceptionRecord
    from app.models.merchant import Merchant


class InvestigationReport(Base):
    __tablename__ = "investigation_reports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False
    )
    exception_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("exceptions.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="COMPLETED")
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="local_ml")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="reconpilot-rf-v1")
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0")
    observed_discrepancy: Mapped[str] = mapped_column(Text, nullable=False)
    report_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    cited_evidence_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    requires_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    policy_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    evidence_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    evidence_snapshot_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    ranked_alternatives: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    inference_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    merchant: Mapped["Merchant"] = relationship("Merchant")
    exception: Mapped["ExceptionRecord"] = relationship(
        "ExceptionRecord", back_populates="investigation_reports"
    )

    __table_args__ = (
        Index("ix_investigation_reports_merchant_id", "merchant_id"),
        Index("ix_investigation_reports_exception_id", "exception_id"),
        Index("ix_investigation_reports_status", "status"),
        Index("ix_investigation_reports_created_at", "created_at"),
    )
