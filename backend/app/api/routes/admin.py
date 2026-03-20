"""
Admin Routes
─────────────
GET  /api/v1/admin/stats          — system statistics
GET  /api/v1/admin/models         — all model versions
POST /api/v1/admin/models/activate — activate a model version
GET  /api/v1/admin/predictions    — all predictions (admin view)
POST /api/v1/admin/retrain        — manually trigger retraining
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, require_role
from app.models.database_models import (
    ModelVersion, Prediction, PredictionFeedback, User, UserRole
)
from app.schemas.prediction_schemas import ModelVersionResponse, SystemStatsResponse

router = APIRouter()
admin_only = require_role(UserRole.ADMIN)


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
):
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    total = await db.scalar(select(func.count()).select_from(Prediction))
    today = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.created_at >= day_ago)
    )
    week = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.created_at >= week_ago)
    )
    unique_users = await db.scalar(
        select(func.count(func.distinct(Prediction.user_id)))
        .where(Prediction.created_at >= day_ago)
    )
    avg_conf = await db.scalar(select(func.avg(Prediction.confidence)))
    avg_ms = await db.scalar(select(func.avg(Prediction.processing_ms)))

    # Feedback accuracy
    total_fb = await db.scalar(select(func.count()).select_from(PredictionFeedback))
    correct_fb = await db.scalar(
        select(func.count()).select_from(PredictionFeedback)
        .where(PredictionFeedback.is_correct == True)
    )
    feedback_rate = (correct_fb / total_fb) if total_fb else 0.0

    # Active models
    active_models = await db.execute(
        select(ModelVersion.model_name).where(ModelVersion.is_active == True)
    )
    active_model_names = [row[0] for row in active_models.all()]

    return SystemStatsResponse(
        total_predictions=total or 0,
        predictions_today=today or 0,
        predictions_this_week=week or 0,
        unique_users_today=unique_users or 0,
        top_diseases=[],   # TODO: aggregate top N disease_codes
        avg_confidence=float(avg_conf or 0),
        avg_processing_ms=float(avg_ms or 0),
        model_versions_active=active_model_names,
        correct_feedback_rate=float(feedback_rate),
    )


@router.get("/models", response_model=List[ModelVersionResponse])
async def list_model_versions(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
):
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    return result.scalars().all()


@router.post("/models/{model_id}/activate")
async def activate_model(
    model_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(admin_only),
):
    """Swap the active production model to a different version."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(404, "Model version not found")

    # Deactivate current active model
    await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    # (simplified — in prod use UPDATE SET)

    mv.is_active = True
    mv.deployed_at = datetime.now(timezone.utc)
    await db.commit()

    # Hot-swap in model manager
    model_manager = request.app.state.model_manager
    await model_manager.swap_primary_model(mv.model_name, mv.version)

    return {"message": f"Model {mv.model_name} v{mv.version} is now active."}


@router.post("/retrain")
async def trigger_retraining(
    _: User = Depends(admin_only),
):
    """Manually trigger model retraining via Celery."""
    from app.tasks.prediction_tasks import trigger_retraining as retrain_task
    task = retrain_task.delay(feedback_threshold=0)   # 0 = force regardless of count
    return {"message": "Retraining job submitted.", "task_id": task.id}
