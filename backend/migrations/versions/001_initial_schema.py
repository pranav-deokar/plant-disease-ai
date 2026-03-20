"""initial schema

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01 00:00:00

Creates all core tables:
  - users
  - crop_diseases
  - treatment_records
  - predictions
  - prediction_feedback
  - api_keys
  - model_versions
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ── users ──────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(200)),
        sa.Column("role", sa.Enum("farmer","agronomist","admin","api_client", name="userrole"), nullable=False, server_default="farmer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("phone_number", sa.String(20)),
        sa.Column("location", sa.String(200)),
        sa.Column("farm_size_ha", sa.Float()),
        sa.Column("primary_crops", postgresql.ARRAY(sa.String())),
        sa.Column("preferences", postgresql.JSON(), server_default="{}"),
        sa.Column("last_login_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_username", "users", ["username"])

    # ── crop_diseases ──────────────────────────────────────────────────────────
    op.create_table(
        "crop_diseases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("disease_code", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(200), nullable=False),
        sa.Column("scientific_name", sa.String(200)),
        sa.Column("crop_name", sa.String(100), nullable=False),
        sa.Column("is_healthy", sa.Boolean(), server_default="false"),
        sa.Column("pathogen_type", sa.String(50)),
        sa.Column("severity_default", sa.Enum("healthy","mild","moderate","severe","critical", name="severitylevel"), server_default="moderate"),
        sa.Column("is_contagious", sa.Boolean(), server_default="true"),
        sa.Column("spread_rate", sa.String(20)),
        sa.Column("economic_impact", sa.String(20)),
        sa.Column("knowledge_doc_id", sa.String(50)),
        sa.Column("class_index", sa.Integer(), nullable=False),
        sa.Column("training_samples", sa.Integer(), server_default="0"),
        sa.Column("model_accuracy", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("disease_code"),
        sa.UniqueConstraint("class_index"),
    )
    op.create_index("ix_crop_diseases_crop_name", "crop_diseases", ["crop_name"])

    # ── treatment_records ──────────────────────────────────────────────────────
    op.create_table(
        "treatment_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("disease_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("treatment_type", sa.String(30), nullable=False),
        sa.Column("treatment_name", sa.String(200), nullable=False),
        sa.Column("active_ingredient", sa.String(200)),
        sa.Column("application_method", sa.Text()),
        sa.Column("dosage", sa.String(200)),
        sa.Column("frequency", sa.String(100)),
        sa.Column("waiting_period_days", sa.Integer()),
        sa.Column("efficacy_score", sa.Float()),
        sa.Column("cost_level", sa.String(20)),
        sa.Column("availability", sa.String(20)),
        sa.Column("notes", sa.Text()),
        sa.Column("source_reference", sa.String(500)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["disease_id"], ["crop_diseases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_treatment_records_disease_id", "treatment_records", ["disease_id"])

    # ── predictions ────────────────────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True)),
        sa.Column("image_key", sa.String(500), nullable=False),
        sa.Column("image_url", sa.String(1000)),
        sa.Column("image_hash", sa.String(64)),
        sa.Column("original_filename", sa.String(255)),
        sa.Column("image_size_bytes", sa.Integer()),
        sa.Column("image_width", sa.Integer()),
        sa.Column("image_height", sa.Integer()),
        sa.Column("metadata", postgresql.JSON(), server_default="{}"),
        sa.Column("status", sa.Enum("pending","processing","completed","failed", name="predictionstatus"), server_default="pending"),
        sa.Column("model_name", sa.String(100)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("processing_ms", sa.Integer()),
        sa.Column("disease_id", postgresql.UUID(as_uuid=True)),
        sa.Column("confidence", sa.Float()),
        sa.Column("severity", sa.Enum("healthy","mild","moderate","severe","critical", name="severitylevel")),
        sa.Column("severity_score", sa.Float()),
        sa.Column("top_predictions", postgresql.JSON(), server_default="[]"),
        sa.Column("gradcam_key", sa.String(500)),
        sa.Column("gradcam_url", sa.String(1000)),
        sa.Column("attention_regions", postgresql.JSON()),
        sa.Column("error_message", sa.Text()),
        sa.Column("retry_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["disease_id"], ["crop_diseases.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])
    op.create_index("ix_predictions_status", "predictions", ["status"])
    op.create_index("ix_predictions_image_hash", "predictions", ["image_hash"])

    # ── prediction_feedback ────────────────────────────────────────────────────
    op.create_table(
        "prediction_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("correct_disease_code", sa.String(100)),
        sa.Column("user_notes", sa.Text()),
        sa.Column("treatment_helpful", sa.Boolean()),
        sa.Column("used_for_retraining", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id"),
    )

    # ── api_keys ───────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("scopes", postgresql.ARRAY(sa.String()), server_default="{}"),
        sa.Column("rate_limit_rpm", sa.Integer(), server_default="60"),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("request_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )

    # ── model_versions ─────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("architecture", sa.String(100)),
        sa.Column("mlflow_run_id", sa.String(100)),
        sa.Column("artifact_uri", sa.String(500)),
        sa.Column("num_classes", sa.Integer()),
        sa.Column("input_size", sa.Integer()),
        sa.Column("val_accuracy", sa.Float()),
        sa.Column("val_f1_macro", sa.Float()),
        sa.Column("val_loss", sa.Float()),
        sa.Column("test_accuracy", sa.Float()),
        sa.Column("per_class_metrics", postgresql.JSON()),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("is_shadow", sa.Boolean(), server_default="false"),
        sa.Column("deployed_at", sa.DateTime(timezone=True)),
        sa.Column("deprecated_at", sa.DateTime(timezone=True)),
        sa.Column("deployment_notes", sa.Text()),
        sa.Column("total_predictions", sa.Integer(), server_default="0"),
        sa.Column("avg_confidence", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])


def downgrade():
    op.drop_table("model_versions")
    op.drop_table("api_keys")
    op.drop_table("prediction_feedback")
    op.drop_table("predictions")
    op.drop_table("treatment_records")
    op.drop_table("crop_diseases")
    op.drop_table("users")
    # Drop enums
    for name in ["userrole", "predictionstatus", "severitylevel"]:
        op.execute(f"DROP TYPE IF EXISTS {name}")
