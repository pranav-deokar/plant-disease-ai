"""
Pydantic v2 schemas for API request/response validation.
These are the shapes the API serializes to and from JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class TopPredictionSchema(BaseModel):
    rank: int
    disease_code: str
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class AttentionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction_id: str
    disease_code: str
    disease_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str
    severity_score: float = Field(..., ge=0.0, le=1.0)
    top_k: List[TopPredictionSchema]
    gradcam_url: Optional[str] = None
    attention_boxes: List[Dict[str, Any]] = []
    image_quality_score: float
    is_leaf_detected: bool
    warnings: List[str] = []
    processing_ms: int
    treatments: Dict[str, Any] = {}
    model_name: str
    model_version: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PredictionListResponse(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class PredictionFeedbackRequest(BaseModel):
    is_correct: bool
    correct_disease_code: Optional[str] = None
    user_notes: Optional[str] = Field(None, max_length=1000)
    treatment_helpful: Optional[bool] = None

    @field_validator("correct_disease_code")
    @classmethod
    def validate_disease_code(cls, v, info):
        if not info.data.get("is_correct") and v is None:
            raise ValueError("correct_disease_code is required when is_correct=False")
        return v


class PredictionFeedbackResponse(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID
    is_correct: bool
    correct_disease_code: Optional[str]
    treatment_helpful: Optional[bool]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    primary_crops: List[str] = []


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds


class UserProfileResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    location: Optional[str]
    farm_size_ha: Optional[float]
    primary_crops: Optional[List[str]]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Admin ─────────────────────────────────────────────────────────────────────

class SystemStatsResponse(BaseModel):
    total_predictions: int
    predictions_today: int
    predictions_this_week: int
    unique_users_today: int
    top_diseases: List[Dict[str, Any]]   # [{disease_name, count, percentage}]
    avg_confidence: float
    avg_processing_ms: float
    model_versions_active: List[str]
    correct_feedback_rate: float


class ModelVersionResponse(BaseModel):
    id: uuid.UUID
    model_name: str
    version: str
    architecture: str
    val_accuracy: Optional[float]
    val_f1_macro: Optional[float]
    is_active: bool
    is_shadow: bool
    deployed_at: Optional[datetime]
    total_predictions: int
    avg_confidence: Optional[float]

    model_config = {"from_attributes": True}
