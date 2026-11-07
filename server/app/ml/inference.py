"""
Local ML Inference Engine for ReconPilot Root-Cause Investigation.
Executes CPU-based model inference, computes class probabilities,
applies confidence/margin abstention gates, and produces ranked hypotheses.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.ml.model_registry import ModelRegistry, get_model_registry


@dataclass
class PredictionResult:
    predicted_cause: str
    confidence_score: float
    ranked_alternatives: list[dict[str, Any]]
    is_abstaining: bool
    abstention_reason: str | None
    inference_mode: str
    model_name: str


class LocalMLInferenceService:
    """
    Performs inference with conservative confidence thresholding and abstention.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_model_registry()
        self.settings = get_settings()

    def predict(self, feature_dict: dict[str, Any]) -> PredictionResult:
        """
        Evaluates a feature vector and returns a structured PredictionResult.
        """
        if self.registry.is_degraded or self.registry.pipeline is None:
            # Degraded mode: fallback to undetermined / review required
            return PredictionResult(
                predicted_cause="INSUFFICIENT_EVIDENCE",
                confidence_score=0.0,
                ranked_alternatives=[],
                is_abstaining=True,
                abstention_reason=self.registry.load_error or "Local ML model not available.",
                inference_mode="DETERMINISTIC_FALLBACK",
                model_name="deterministic-fallback",
            )

        pipeline = self.registry.pipeline
        classes = list(pipeline.classes_)

        df_single = pd.DataFrame([feature_dict])

        try:
            probabilities = pipeline.predict_proba(df_single)[0]
        except Exception as e:
            return PredictionResult(
                predicted_cause="INSUFFICIENT_EVIDENCE",
                confidence_score=0.0,
                ranked_alternatives=[],
                is_abstaining=True,
                abstention_reason=f"Model inference exception: {e}",
                inference_mode="DETERMINISTIC_FALLBACK",
                model_name=self.registry.model_name,
            )

        # Sort classes by descending probability
        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = sorted_indices[0]
        top_cause = classes[top_idx]
        top_prob = float(probabilities[top_idx])

        ranked_alternatives = [
            {"cause": str(classes[idx]), "probability": round(float(probabilities[idx]), 4)}
            for idx in sorted_indices[:4]
        ]

        conf_threshold = self.settings.ML_CONFIDENCE_THRESHOLD
        min_margin = self.settings.ML_MIN_MARGIN

        # Check abstention conditions
        is_abstaining = False
        abstention_reason = None

        if top_prob < conf_threshold:
            is_abstaining = True
            abstention_reason = f"Model confidence ({top_prob:.2f}) is below the required conservative threshold ({conf_threshold:.2f})."
            top_cause = "INSUFFICIENT_EVIDENCE"

        elif len(sorted_indices) > 1:
            second_prob = float(probabilities[sorted_indices[1]])
            margin = top_prob - second_prob
            if margin < min_margin:
                is_abstaining = True
                abstention_reason = (
                    f"Ambiguous prediction: top candidate '{top_cause}' ({top_prob:.2f}) is too close to "
                    f"'{classes[sorted_indices[1]]}' ({second_prob:.2f}) with margin {margin:.2f} < {min_margin:.2f}."
                )
                top_cause = "AMBIGUOUS_MATCH"

        return PredictionResult(
            predicted_cause=top_cause,
            confidence_score=round(top_prob, 4),
            ranked_alternatives=ranked_alternatives,
            is_abstaining=is_abstaining,
            abstention_reason=abstention_reason,
            inference_mode="LOCAL_ML",
            model_name=self.registry.model_name,
        )
