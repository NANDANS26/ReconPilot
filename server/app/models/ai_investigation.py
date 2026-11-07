import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
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
from app.models.enums import AIClassification, ValidationStatus

if TYPE_CHECKING:
    from app.models.exception import ExceptionRecord


class AIInvestigation(Base):
    __tablename__ = "ai_investigations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exception_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("exceptions.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="local_ml")
    model: Mapped[str] = mapped_column(String(100), nullable=False, default="reconpilot-rf-v1")
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0")
    classification: Mapped[AIClassification] = mapped_column(
        Enum(AIClassification, name="ai_classification_enum", native_enum=False),
        nullable=False,
    )
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    recommended_action: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence_references: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    validation_status: Mapped[ValidationStatus] = mapped_column(
        Enum(ValidationStatus, name="validation_status_enum", native_enum=False),
        nullable=False,
        default=ValidationStatus.VALID,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    exception: Mapped["ExceptionRecord"] = relationship(
        "ExceptionRecord", back_populates="ai_investigations"
    )

    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="ck_ai_investigations_confidence_range"
        ),
        Index("ix_ai_investigations_exception_id", "exception_id"),
        Index("ix_ai_investigations_classification", "classification"),
    )
