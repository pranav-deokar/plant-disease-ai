"""
User Routes
────────────
GET  /api/v1/users/me            — get own profile
PATCH /api/v1/users/me           — update profile
GET  /api/v1/users/me/stats      — prediction statistics for current user
POST /api/v1/users/me/api-keys   — create API key
GET  /api/v1/users/me/api-keys   — list API keys
DELETE /api/v1/users/me/api-keys/{key_id} — revoke API key
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.models.database_models import APIKey, Prediction, User
from app.schemas.prediction_schemas import UserProfileResponse

router = APIRouter()


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    farm_size_ha: Optional[float] = None
    primary_crops: Optional[List[str]] = None


class APIKeyCreateRequest(BaseModel):
    name: str
    scopes: List[str] = ["predict", "read"]
    rate_limit_rpm: int = 60


@router.get("/me", response_model=UserProfileResponse)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserProfileResponse)
async def update_profile(
    body: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(current_user, field, value)
    current_user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.get("/me/stats")
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import case
    from app.models.database_models import CropDisease

    total = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.user_id == current_user.id)
    )
    avg_conf = await db.scalar(
        select(func.avg(Prediction.confidence)).where(Prediction.user_id == current_user.id)
    )

    # Top 5 detected diseases
    top_diseases_q = (
        select(CropDisease.display_name, func.count().label("count"))
        .join(Prediction, Prediction.disease_id == CropDisease.id)
        .where(Prediction.user_id == current_user.id, CropDisease.is_healthy == False)
        .group_by(CropDisease.display_name)
        .order_by(func.count().desc())
        .limit(5)
    )
    top_result = await db.execute(top_diseases_q)
    top_diseases = [{"disease": row[0], "count": row[1]} for row in top_result.all()]

    return {
        "total_predictions": total or 0,
        "avg_confidence": round(float(avg_conf or 0), 4),
        "top_diseases": top_diseases,
        "member_since": current_user.created_at.isoformat(),
    }


@router.post("/me/api-keys")
async def create_api_key(
    body: APIKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Check existing key count
    count = await db.scalar(
        select(func.count()).select_from(APIKey)
        .where(APIKey.user_id == current_user.id, APIKey.is_active == True)
    )
    if count >= 10:
        raise HTTPException(400, "Maximum of 10 active API keys allowed.")

    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = APIKey(
        user_id=current_user.id,
        key_hash=key_hash,
        name=body.name,
        scopes=body.scopes,
        rate_limit_rpm=body.rate_limit_rpm,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return {
        "id": str(api_key.id),
        "name": api_key.name,
        "key": raw_key,  # Only shown once — user must store it
        "scopes": api_key.scopes,
        "created_at": api_key.created_at.isoformat(),
        "warning": "Store this key securely. It will not be shown again.",
    }


@router.get("/me/api-keys")
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == current_user.id, APIKey.is_active == True)
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        {
            "id": str(k.id),
            "name": k.name,
            "scopes": k.scopes,
            "rate_limit_rpm": k.rate_limit_rpm,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            "request_count": k.request_count,
            "created_at": k.created_at.isoformat(),
        }
        for k in keys
    ]


@router.delete("/me/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user.id)
    )
    key = result.scalar_one_or_none()
    if key is None:
        raise HTTPException(404, "API key not found.")
    key.is_active = False
    await db.commit()
