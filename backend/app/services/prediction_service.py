"""
Prediction Service
───────────────────
Orchestrates the full inference pipeline:
  1. Preprocess image
  2. Run model inference
  3. Compute Grad-CAM visualization
  4. Estimate disease severity
  5. Fetch treatment recommendations
  6. Persist result to DB + S3
  7. Return structured PredictionResponse
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import torch
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.ml.models.model_manager import ModelManager, DISEASE_CLASSES, NUM_CLASSES
from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
from app.ml.explainability.gradcam import GradCAMPlusPlus, encode_overlay_to_bytes
from app.services.storage_service import StorageService
from app.services.advisory_service import AdvisoryService
from app.models.database_models import Prediction, PredictionStatus, SeverityLevel

logger = logging.getLogger(__name__)


@dataclass
class TopPrediction:
    disease_code: str
    display_name: str
    confidence: float
    rank: int


@dataclass
class PredictionOutput:
    prediction_id: str
    disease_code: str
    disease_name: str
    confidence: float
    severity: str
    severity_score: float
    top_k: List[TopPrediction]
    _url: Optional[str]
    attention_boxes: list
    image_quality_score: float
    is_leaf_detected: bool
    warnings: List[str]
    processing_ms: int
    treatments: dict
    model_name: str
    model_version: str


SEVERITY_THRESHOLDS = {
    # (coverage_ratio, confidence) → SeverityLevel
    # Coverage = fraction of leaf flagged as diseased by Grad-CAM
    "healthy": (0.0, 0.0),
    "mild":    (0.05, 0.50),
    "moderate":(0.20, 0.65),
    "severe":  (0.40, 0.80),
    "critical":(0.65, 0.90),
}


class PredictionService:
    """
    Stateless service — inject one per request via FastAPI dependency.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageService,
        advisory: AdvisoryService,
        db: AsyncSession,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.advisory = advisory
        self.db = db
        self.preprocessor = ImagePreprocessor()

    async def predict(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        crop_hint: Optional[str] = None,
        top_k: int = 5,
        recommendation_mode: str = "db",
    ) -> PredictionOutput:
        """
        End-to-end prediction pipeline.
        Returns a fully populated PredictionOutput.
        """
        start_time = time.perf_counter()
        prediction_id = str(uuid.uuid4())
        warnings = []

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        try:
            prep_result = await self.preprocessor.preprocess_for_inference(
                image_bytes,
                apply_enhancement=False,
                segment_leaf=False,
            )
        except ValueError as e:
            raise ValueError(f"Image preprocessing failed: {e}")

        warnings.extend(prep_result.warnings)

        # ── Step 2: Model Inference ───────────────────────────────────────────
        loaded_model = self._select_model(crop_hint)
        tensor = prep_result.tensor  # already on correct device

        with torch.no_grad():
            logits = loaded_model.model(tensor)
            logits = loaded_model.model(tensor)
            temperature = 0.8
            probs = torch.softmax(logits / temperature, dim=-1).squeeze()
        probs_np = probs.cpu().numpy()
        top_indices = probs_np.argsort()[::-1][:top_k]

        primary_idx = int(top_indices[0])
        primary_confidence = float(probs_np[primary_idx])
        primary_disease_code = DISEASE_CLASSES[primary_idx]

        # Low confidence warning
        if primary_confidence < settings.MODEL_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Prediction confidence is low ({primary_confidence:.0%}). "
                "Please upload a clearer image of the leaf for better results."
            )

        top_predictions = [
            TopPrediction(
                disease_code=DISEASE_CLASSES[idx],
                display_name=self._format_disease_name(DISEASE_CLASSES[idx]),
                confidence=float(probs_np[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

        # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
       # ── Step 3: Grad-CAM (DISABLED) ─────────────────────────────────────────
        gradcam_url = None
        attention_boxes = []
        coverage_ratio = 0.0

        # ── Step 4: Severity Estimation ───────────────────────────────────────
        is_healthy = primary_disease_code.endswith("healthy")
        severity, severity_score = self._estimate_severity(
            is_healthy, primary_confidence, coverage_ratio
        )

        # ── Step 5: Treatment Recommendations ────────────────────────────────
        treatments = {}
        if not is_healthy:
            try:
                treatments = await self.advisory.get_treatments(
                    primary_disease_code,
                    mode=recommendation_mode 
                )
            except Exception as e:
                logger.warning(f"Advisory service failed: {e}")
                treatments = {"error": "Treatment recommendations temporarily unavailable."}

        # ── Step 6: Upload original image ─────────────────────────────────────
        image_key = f"uploads/{prediction_id}/original{self._get_ext(original_filename)}"
        image_url = None

        # ── Step 7: Persist to DB ─────────────────────────────────────────────
        processing_ms = int((time.perf_counter() - start_time) * 1000)
        await self._save_prediction(
            prediction_id=prediction_id,
            user_id=user_id,
            image_key=image_key,
            image_url=image_url,
            image_hash=prep_result.image_hash,
            original_filename=original_filename,
            image_size_bytes=len(image_bytes),
            image_width=prep_result.width,
            image_height=prep_result.height,
            disease_code=primary_disease_code,
            confidence=primary_confidence,
            severity=severity,
            severity_score=severity_score,
            top_predictions=[
                {"disease_code": p.disease_code, "confidence": p.confidence}
                for p in top_predictions
            ],
            gradcam_key=gradcam_key if gradcam_url else None,
            gradcam_url=gradcam_url,
            attention_regions=attention_boxes,
            model_name=loaded_model.name,
            model_version=loaded_model.version,
            processing_ms=processing_ms,
            
        )

        self.model_manager.record_inference(loaded_model.name, float(processing_ms))

        return PredictionOutput(
            prediction_id=prediction_id,
            disease_code=primary_disease_code,
            disease_name=self._format_disease_name(primary_disease_code),
            confidence=primary_confidence,
            severity=severity.value,
            severity_score=severity_score,
            top_k=top_predictions,
            
            attention_boxes=attention_boxes,
            image_quality_score=prep_result.quality_score,
            is_leaf_detected=prep_result.is_leaf_detected,
            warnings=warnings,
            processing_ms=processing_ms,
            treatments=treatments,
            model_name=loaded_model.name,
            model_version=loaded_model.version,
        )

    def _select_model(self, crop_hint: Optional[str]):
        """
        Select model. In future: crop-specific fine-tuned models.
        For now: always use primary; fall back if primary inference fails.
        """
        return self.model_manager.get_primary_model()

    def _estimate_severity(
        self,
        is_healthy: bool,
        confidence: float,
        coverage_ratio: float,
    ) -> tuple[SeverityLevel, float]:
        if is_healthy:
            return SeverityLevel.HEALTHY, 0.0

        # Combine coverage and confidence into a severity score [0, 1]
        severity_score = min(0.6 * coverage_ratio + 0.4 * confidence, 1.0)

        if severity_score < 0.15:
            return SeverityLevel.MILD, severity_score
        elif severity_score < 0.40:
            return SeverityLevel.MODERATE, severity_score
        elif severity_score < 0.70:
            return SeverityLevel.SEVERE, severity_score
        else:
            return SeverityLevel.CRITICAL, severity_score

    def _format_disease_name(self, disease_code: str) -> str:
        """'tomato___early_blight' → 'Tomato - Early Blight'"""
        parts = disease_code.split("___")
        if len(parts) == 2:
            crop, disease = parts
            return f"{crop.replace('_', ' ').title()} — {disease.replace('_', ' ').title()}"
        return disease_code.replace("_", " ").title()

    def _get_ext(self, filename: Optional[str]) -> str:
        if filename and "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ".jpg"

    async def _save_prediction(self, **kwargs):
        """Persist prediction record to PostgreSQL."""
        from app.models.database_models import CropDisease
        from sqlalchemy import select

        disease_code = kwargs.pop("disease_code")

        # Look up disease FK
        result = await self.db.execute(
            select(CropDisease).where(CropDisease.disease_code == disease_code)
        )
        disease = result.scalar_one_or_none()

        record = Prediction(
            id=uuid.UUID(kwargs["prediction_id"]),
            user_id=uuid.UUID(kwargs["user_id"]) if kwargs.get("user_id") else None,
            disease_id=disease.id if disease else None,
            status=PredictionStatus.COMPLETED,
            **{k: v for k, v in kwargs.items() if k != "prediction_id" and k != "user_id"},
        )
        self.db.add(record)
        await self.db.commit()
