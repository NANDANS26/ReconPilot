"""
ReconPilot Local Machine Learning Engine for Root-Cause Investigation.
Provides reproducible feature extraction, model registry, local CPU inference,
and deterministic evidence-grounded explanation generation.
"""

from app.ml.feature_schema import FEATURE_SCHEMA_VERSION, TARGET_LABELS
from app.ml.model_registry import get_model_registry

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "TARGET_LABELS",
    "get_model_registry",
]
