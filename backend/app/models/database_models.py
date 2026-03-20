"""
SQLAlchemy ORM models for PostgreSQL.
All tables use UUIDs as primary keys and track created_at / updated_at timestamps.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base


# ── Enums ──────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    FARMER = "farmer"
    AGRONOMIST = "agronomist"
    ADMIN = "admin"
    API_CLIENT = "api_client"

class PredictionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SeverityLevel(str, enum.Enum):
    HEALTHY = "healthy"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# ── Mixins ────────────────────────────────────────────────────────────────────

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now(), nullable=False)

class UUIDMixin:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)


# ── Tables ────────────────────────────────────────────────────────────────────

class User(Base, UUIDMixin, TimestampMixin):
    """System users — farmers, agronomists, admins."""
    __tablename__ = "users"

    email           = Column(String(255), unique=True, index=True, nullable=False)
    username        = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(200))
    role            = Column(SAEnum(UserRole), default=UserRole.FARMER, nullable=False)
    is_active       = Column(Boolean, default=True, nullable=False)
    is_verified     = Column(Boolean, default=False, nullable=False)
    phone_number    = Column(String(20))
    location        = Column(String(200))          # "State, Country"
    farm_size_ha    = Column(Float)
    primary_crops   = Column(ARRAY(String))        # ["tomato", "potato", "maize"]
    preferences     = Column(JSON, default=dict)   # UI/notification prefs
    last_login_at   = Column(DateTime(timezone=True))

    # Relationships
    predictions     = relationship("Prediction", back_populates="user", lazy="dynamic")
    api_keys        = relationship("APIKey", back_populates="user")
    feedback        = relationship("PredictionFeedback", back_populates="user")


class CropDisease(Base, UUIDMixin, TimestampMixin):
    """
    Master disease catalogue.
    Rich data lives in MongoDB; this table holds the structured subset
    needed for joins and filtering (SQL-friendly).
    """
    __tablename__ = "crop_diseases"

    # Identity
    disease_code    = Column(String(100), unique=True, index=True, nullable=False)
    # e.g. "tomato_early_blight" — matches PlantVillage class label
    display_name    = Column(String(200), nullable=False)
    scientific_name = Column(String(200))
    crop_name       = Column(String(100), nullable=False, index=True)
    is_healthy      = Column(Boolean, default=False)     # "Healthy" pseudo-class

    # Classification metadata
    pathogen_type   = Column(String(50))    # fungal | bacterial | viral | pest | nutritional
    severity_default= Column(SAEnum(SeverityLevel), default=SeverityLevel.MODERATE)
    is_contagious   = Column(Boolean, default=True)
    spread_rate     = Column(String(20))    # slow | moderate | fast | explosive
    economic_impact = Column(String(20))    # low | medium | high | critical

    # MongoDB reference for full knowledge article
    knowledge_doc_id = Column(String(50))  # MongoDB ObjectId as string

    # ML metadata
    class_index     = Column(Integer, unique=True, nullable=False)  # model output index
    training_samples= Column(Integer, default=0)
    model_accuracy  = Column(Float)         # per-class accuracy from eval

    # Relationships
    predictions     = relationship("Prediction", back_populates="disease")
    treatments      = relationship("TreatmentRecord", back_populates="disease")


class TreatmentRecord(Base, UUIDMixin, TimestampMixin):
    """
    Structured treatment recommendations.
    One disease can have many treatment records (organic, chemical, cultural).
    """
    __tablename__ = "treatment_records"

    disease_id          = Column(UUID(as_uuid=True), ForeignKey("crop_diseases.id"), nullable=False, index=True)
    treatment_type      = Column(String(30), nullable=False)  # chemical | organic | cultural | biological
    treatment_name      = Column(String(200), nullable=False)
    active_ingredient   = Column(String(200))
    application_method  = Column(Text)
    dosage              = Column(String(200))
    frequency           = Column(String(100))
    waiting_period_days = Column(Integer)        # harvest safety interval
    efficacy_score      = Column(Float)          # 0-1, from literature
    cost_level          = Column(String(20))     # low | medium | high
    availability        = Column(String(20))     # common | specialty | prescription
    notes               = Column(Text)
    source_reference    = Column(String(500))    # citation URL / paper

    # Relationships
    disease = relationship("CropDisease", back_populates="treatments")


class Prediction(Base, UUIDMixin, TimestampMixin):
    """
    Each image submission creates one prediction record.
    Supports async processing via status field.
    """
    __tablename__ = "predictions"

    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    # nullable — anonymous predictions allowed

    # Input
    image_key       = Column(String(500), nullable=False)  # S3 object key
    image_url       = Column(String(1000))                 # pre-signed URL (short TTL)
    image_hash      = Column(String(64), index=True)       # SHA-256 for dedup
    original_filename = Column(String(255))
    image_size_bytes = Column(Integer)
    image_width     = Column(Integer)
    image_height    = Column(Integer)
    extra_data      = Column(JSON, default=dict)           # GPS, device, crop_hint
    # Processing
    status          = Column(SAEnum(PredictionStatus), default=PredictionStatus.PENDING, index=True)
    model_name      = Column(String(100))
    model_version   = Column(String(50))
    processing_ms   = Column(Integer)      # inference time in milliseconds

    # Primary result
    disease_id      = Column(UUID(as_uuid=True), ForeignKey("crop_diseases.id"), nullable=True, index=True)
    confidence      = Column(Float)        # 0-1
    severity        = Column(SAEnum(SeverityLevel))
    severity_score  = Column(Float)        # 0-1 continuous severity

    # Top-k alternatives (JSON array of {disease_code, confidence})
    top_predictions = Column(JSON, default=list)

    # Explainability
    gradcam_key     = Column(String(500))  # S3 key for Grad-CAM heatmap image
    gradcam_url     = Column(String(1000))
    attention_regions = Column(JSON)       # bounding boxes of disease regions

    # Error handling
    error_message   = Column(Text)
    retry_count     = Column(Integer, default=0)

    # Relationships
    user        = relationship("User", back_populates="predictions")
    disease     = relationship("CropDisease", back_populates="predictions")
    feedback    = relationship("PredictionFeedback", back_populates="prediction", uselist=False)


class PredictionFeedback(Base, UUIDMixin, TimestampMixin):
    """User feedback on prediction accuracy — used for model retraining."""
    __tablename__ = "prediction_feedback"

    prediction_id       = Column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False, unique=True)
    user_id             = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_correct          = Column(Boolean, nullable=False)
    correct_disease_code= Column(String(100))   # if prediction was wrong
    user_notes          = Column(Text)
    treatment_helpful   = Column(Boolean)
    used_for_retraining = Column(Boolean, default=False)

    prediction  = relationship("Prediction", back_populates="feedback")
    user        = relationship("User", back_populates="feedback")


class APIKey(Base, UUIDMixin, TimestampMixin):
    """API keys for programmatic access (mobile apps, third-party integrations)."""
    __tablename__ = "api_keys"

    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    key_hash        = Column(String(64), unique=True, nullable=False)   # SHA-256 of raw key
    name            = Column(String(100), nullable=False)               # human label
    is_active       = Column(Boolean, default=True)
    scopes          = Column(ARRAY(String), default=list)               # ["predict", "read"]
    rate_limit_rpm  = Column(Integer, default=60)
    last_used_at    = Column(DateTime(timezone=True))
    expires_at      = Column(DateTime(timezone=True))
    request_count   = Column(Integer, default=0)

    user = relationship("User", back_populates="api_keys")


class ModelVersion(Base, UUIDMixin, TimestampMixin):
    """Registry of trained model versions for deployment tracking."""
    __tablename__ = "model_versions"

    model_name      = Column(String(100), nullable=False, index=True)
    version         = Column(String(50), nullable=False)
    architecture    = Column(String(100))      # EfficientNet-B4 | MobileNetV3 etc.
    mlflow_run_id   = Column(String(100))
    artifact_uri    = Column(String(500))      # S3 path to model weights
    num_classes     = Column(Integer)
    input_size      = Column(Integer)

    # Evaluation metrics
    val_accuracy    = Column(Float)
    val_f1_macro    = Column(Float)
    val_loss        = Column(Float)
    test_accuracy   = Column(Float)
    per_class_metrics = Column(JSON)           # {class_name: {precision, recall, f1}}

    # Deployment
    is_active       = Column(Boolean, default=False)
    is_shadow       = Column(Boolean, default=False)   # shadow mode for A/B testing
    deployed_at     = Column(DateTime(timezone=True))
    deprecated_at   = Column(DateTime(timezone=True))
    deployment_notes= Column(Text)
    total_predictions = Column(Integer, default=0)
    avg_confidence  = Column(Float)
