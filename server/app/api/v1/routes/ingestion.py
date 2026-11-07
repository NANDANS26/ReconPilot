from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import require_roles
from app.db.session import get_db
from app.models.enums import UserRole
from app.models.user import User
from app.modules.reconciliation.ingestion import (
    ingest_demo_financial_dataset,
    ingest_payments_batch_idempotent,
)
from app.modules.reconciliation.schemas import (
    BatchIngestPaymentsRequest,
    BatchIngestResponse,
    IngestionSummaryResponse,
)
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/ingestion", tags=["Data Ingestion"])


@router.post("/demo", response_model=ApiResponse[IngestionSummaryResponse])
def ingest_demo_data(
    current_user: Annotated[
        User, Depends(require_roles(UserRole.ADMIN, UserRole.FINANCE_OPERATOR))
    ],
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[IngestionSummaryResponse]:
    """
    Ingests and validates the standardized benchmark and operational demo financial records.
    Idempotent: Safe to execute repeatedly without duplicating existing source records.
    """
    summary = ingest_demo_financial_dataset(
        db=db,
        merchant_id=current_user.merchant_id,
        actor_id=str(current_user.id),
    )
    return ApiResponse.ok(summary)


@router.post("/payments", response_model=ApiResponse[BatchIngestResponse])
def ingest_payments(
    payload: BatchIngestPaymentsRequest,
    current_user: Annotated[
        User, Depends(require_roles(UserRole.ADMIN, UserRole.FINANCE_OPERATOR))
    ],
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[BatchIngestResponse]:
    """
    Batch idempotent ingestion endpoint for payment transactions.
    Detects duplicate and conflicting payloads without failing entire batch.
    """
    result = ingest_payments_batch_idempotent(
        db=db,
        merchant_id=current_user.merchant_id,
        payload=payload,
        actor_id=str(current_user.id),
    )
    return ApiResponse.ok(result)
