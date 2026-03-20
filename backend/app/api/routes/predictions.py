"""
Prediction API Routes
─────────────────────
POST /api/v1/predictions/        — submit image, get disease prediction
GET  /api/v1/predictions/{id}    — get single prediction by ID
GET  /api/v1/predictions/        — list predictions (paginated, user-scoped)
POST /api/v1/predictions/{id}/feedback — submit correctness feedback
DELETE /api/v1/predictions/{id}  — soft-delete (admin or owner)
"""

import uuid
from typing import List, Optional
import traceback
from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Query, Request, UploadFile, status
)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.models.database_models import User, Prediction
from app.schemas.prediction_schemas import (
    PredictionResponse, PredictionListResponse,
    PredictionFeedbackRequest, PredictionFeedbackResponse,
)
from app.services.prediction_service import PredictionService
from app.services.storage_service import StorageService
from app.services.advisory_service import AdvisoryService
from app.api.dependencies import get_current_user, get_optional_user, require_role
from app.ml.models.model_manager import ModelManager

router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def get_prediction_service(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> PredictionService:
    return PredictionService(
        model_manager=get_model_manager(request),
        storage=StorageService(),
        advisory=AdvisoryService(db=db),
        db=db,
    )


# ── POST /predictions ──────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a leaf image for disease prediction",
    description="""
Upload a plant leaf image (JPEG, PNG, WebP — max 10 MB).

The response includes:
- `disease_code` and `disease_name` — identified disease
- `confidence` — model confidence (0–1)
- `severity` — healthy / mild / moderate / severe / critical
- `top_k` — top 5 alternative predictions
- `gradcam_url` — Grad-CAM overlay image showing disease regions
- `treatments` — chemical, organic, and cultural treatment options
- `warnings` — image quality or confidence alerts

**Async option**: Set `async_mode=true` for large batches.
The endpoint returns `prediction_id` immediately; poll `GET /predictions/{id}` for results.
""",
)
async def create_prediction(
 
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Plant leaf image"),
    crop_hint: Optional[str] = Form(None, description="Optional crop name to guide prediction (e.g. 'tomato')"),
    top_k: int = Form(5, ge=1, le=10, description="Number of top predictions to return"),
    async_mode: bool = Form(False, description="Return immediately and process asynchronously"),
    recommendation_mode: str = Form("db"),
    current_user: Optional[User] = Depends(get_optional_user),
    service: PredictionService = Depends(get_prediction_service),
    
):
    # Validate file format
    if image.content_type not in {
        "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"
    }:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format: {image.content_type}. "
                   f"Accepted: JPEG, PNG, WebP, BMP.",
        )

    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    user_id = str(current_user.id) if current_user else None

    if async_mode:
        # Kick off async Celery task — return prediction_id immediately
        from app.tasks.prediction_tasks import process_prediction_async
        task = process_prediction_async.delay(
            image_bytes=image_bytes,
            user_id=user_id,
            original_filename=image.filename,
            crop_hint=crop_hint,
            top_k=top_k,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"prediction_id": task.id, "status": "processing"},
        )

    # Synchronous prediction
    try:
        result = await service.predict(
            image_bytes=image_bytes,
            user_id=user_id,
            original_filename=image.filename,
            crop_hint=crop_hint,
            top_k=top_k,
            recommendation_mode=recommendation_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        traceback.print_exc()   # 🔥 THIS WILL SHOW REAL ERROR IN DOCKER LOGS
        raise HTTPException(
            status_code=500,
            detail=str(e)       # optional (for debugging)
        )

    return _to_response(result)


# ── GET /predictions/{id} ──────────────────────────────────────────────────────

@router.get(
    "/{prediction_id}",
    response_model=PredictionResponse,
    summary="Retrieve a prediction by ID",
)
async def get_prediction(
    prediction_id: uuid.UUID,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Prediction)
        .where(Prediction.id == prediction_id)
        .options(selectinload(Prediction.disease), selectinload(Prediction.feedback))
    )
    pred = result.scalar_one_or_none()

    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    # Access control: owner or admin
    if current_user is None or (
        str(pred.user_id) != str(current_user.id) and current_user.role.value != "admin"
    ):
        raise HTTPException(status_code=403, detail="Access denied.")

    return pred  # Pydantic schema handles serialization


# ── GET /predictions ───────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=PredictionListResponse,
    summary="List predictions (user-scoped, paginated)",
)
async def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    crop_name: Optional[str] = Query(None, description="Filter by crop name"),
    disease_code: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select, func
    from sqlalchemy.orm import selectinload
    from app.models.database_models import CropDisease

    query = (
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .options(selectinload(Prediction.disease))
        .order_by(Prediction.created_at.desc())
    )

    if disease_code:
        query = query.where(Prediction.disease.has(CropDisease.disease_code == disease_code))
    if severity:
        query = query.where(Prediction.severity == severity)

    # Count total for pagination
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    predictions = (await db.execute(query)).scalars().all()

    return PredictionListResponse(
        items=predictions,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


# ── POST /predictions/{id}/feedback ───────────────────────────────────────────

@router.post(
    "/{prediction_id}/feedback",
    response_model=PredictionFeedbackResponse,
    summary="Submit feedback on prediction accuracy",
)
async def submit_feedback(
    prediction_id: uuid.UUID,
    feedback: PredictionFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from app.models.database_models import PredictionFeedback

    # Verify ownership
    pred_result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id,
        )
    )
    pred = pred_result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    existing = await db.execute(
        select(PredictionFeedback).where(PredictionFeedback.prediction_id == prediction_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Feedback already submitted for this prediction.")

    fb = PredictionFeedback(
        prediction_id=prediction_id,
        user_id=current_user.id,
        is_correct=feedback.is_correct,
        correct_disease_code=feedback.correct_disease_code,
        user_notes=feedback.user_notes,
        treatment_helpful=feedback.treatment_helpful,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return fb


# ── Helper ─────────────────────────────────────────────────────────────────────

def _to_response(result) -> PredictionResponse:
    return PredictionResponse(
        prediction_id=result.prediction_id,
        disease_code=result.disease_code,
        disease_name=result.disease_name,
        confidence=result.confidence,
        severity=result.severity,
        severity_score=result.severity_score,
        top_k=[
            {
                "rank": p.rank,
                "disease_code": p.disease_code,
                "disease_name": p.display_name,
                "confidence": p.confidence,
            }
            for p in result.top_k
        ],
        gradcam_url=result.gradcam_url,
        attention_boxes=result.attention_boxes,
        image_quality_score=result.image_quality_score,
        is_leaf_detected=result.is_leaf_detected,
        warnings=result.warnings,
        processing_ms=result.processing_ms,
        treatments=result.treatments,
        model_name=result.model_name,
        model_version=result.model_version,
    )
