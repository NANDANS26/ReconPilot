"""
Evidence Snapshot Module for ReconPilot.
Provides immutable, canonicalized, cryptographically-hashed evidence snapshots
capturing the exact state of financial records at the moment of investigation.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from app.modules.evidence.schemas import ExceptionEvidenceResponse

SNAPSHOT_SCHEMA_VERSION = "v1.0"


def canonicalize_snapshot(
    dossier: ExceptionEvidenceResponse,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Serializes an ExceptionEvidenceResponse into a clean, canonical dictionary.
    All Decimal and UUID fields are converted to canonical strings/numbers.
    """
    dossier_dict = dossier.model_dump(mode="json")
    timestamp = captured_at or dossier.reconciliation.created_at or datetime.now(UTC)

    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at": timestamp.isoformat(),
        "exception_id": str(dossier.exception_id),
        "reconciliation": dossier_dict.get("reconciliation", {}),
        "summary": dossier_dict.get("summary", {}),
        "related_records": dossier_dict.get("related_records", {}),
        "evidence_items": dossier_dict.get("evidence_items", []),
    }


def compute_snapshot_hash(snapshot_dict: dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash over the canonical JSON representation
    (with sorted keys and no extraneous whitespace).
    """
    canonical_json = json.dumps(snapshot_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
