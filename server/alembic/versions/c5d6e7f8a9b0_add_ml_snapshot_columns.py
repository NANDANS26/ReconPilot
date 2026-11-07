"""add_ml_snapshot_columns

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-11-07 20:38:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c5d6e7f8a9b0"
down_revision: str | None = "b4c5d6e7f8a9"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "investigation_reports",
        sa.Column("evidence_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("evidence_snapshot_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("model_version", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("feature_schema_version", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("model_score", sa.Numeric(precision=5, scale=4), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("ranked_alternatives", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "investigation_reports",
        sa.Column("inference_mode", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("investigation_reports", "inference_mode")
    op.drop_column("investigation_reports", "ranked_alternatives")
    op.drop_column("investigation_reports", "model_score")
    op.drop_column("investigation_reports", "feature_schema_version")
    op.drop_column("investigation_reports", "model_version")
    op.drop_column("investigation_reports", "evidence_snapshot_hash")
    op.drop_column("investigation_reports", "evidence_snapshot")
