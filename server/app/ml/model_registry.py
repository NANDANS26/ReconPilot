"""
Model Registry for ReconPilot Local ML Investigation Engine.
Loads, validates, and manages local model artifacts with thread-safe caching.
Provides graceful degraded mode if the artifact is missing or corrupted.
"""

import json
import logging
import os
import threading
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.ml.feature_schema import FEATURE_SCHEMA_VERSION, TARGET_LABELS

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry managing the locally persisted RandomForest pipeline.
    """

    _instance: "ModelRegistry | None" = None
    _lock = threading.Lock()

    def __init__(
        self,
        artifact_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self._custom_artifact_path = artifact_path
        self._custom_metadata_path = metadata_path
        self._pipeline: Pipeline | None = None
        self._metadata: dict[str, Any] = {}
        self._is_degraded: bool = False
        self._load_error: str | None = None
        self._load_model()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModelRegistry()
            return cls._instance

    def _resolve_path(self, path: str) -> str:
        """Resolves relative paths against the project root or server directory."""
        if os.path.exists(path):
            return path
        # Try stripping leading 'server/' if running from inside server directory
        if path.startswith("server/") and os.path.exists(path[7:]):
            return path[7:]
        # Try prepending 'server/' if running from project root
        if not path.startswith("server/") and os.path.exists(f"server/{path}"):
            return f"server/{path}"
        return path

    def _load_model(self) -> None:
        """Loads and validates the artifact from disk."""
        artifact_path = self._resolve_path(
            self._custom_artifact_path or self.settings.ML_ARTIFACT_PATH
        )
        metadata_path = self._resolve_path(
            self._custom_metadata_path or self.settings.ML_METADATA_PATH
        )

        if not os.path.exists(artifact_path):
            self._is_degraded = True
            self._load_error = f"Model artifact not found at {artifact_path}. Operating in degraded deterministic mode."
            logger.warning(self._load_error)
            return

        try:
            self._pipeline = joblib.load(artifact_path)
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)

            # Validate feature schema version
            saved_version = self._metadata.get("feature_schema_version")
            if saved_version and saved_version != FEATURE_SCHEMA_VERSION:
                self._is_degraded = True
                self._load_error = (
                    f"Feature schema version mismatch: {saved_version} != {FEATURE_SCHEMA_VERSION}."
                )
                logger.error(self._load_error)
                return

            self._is_degraded = False
            self._load_error = None
            logger.info(f"Loaded local ML model artifact: {artifact_path}")
        except Exception as e:
            self._is_degraded = True
            self._load_error = f"Failed to load ML artifact: {e}"
            logger.error(self._load_error, exc_info=True)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None and not self._is_degraded

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded or self._pipeline is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def pipeline(self) -> Pipeline | None:
        return self._pipeline

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def model_name(self) -> str:
        return self._metadata.get("model_name", self.settings.ML_MODEL_NAME)

    @property
    def classes(self) -> list[str]:
        return self._metadata.get("classes", TARGET_LABELS)

    def reload(self) -> None:
        """Forces a reload of the model from disk."""
        with self._lock:
            self._load_model()


def get_model_registry() -> ModelRegistry:
    return ModelRegistry.get_instance()
