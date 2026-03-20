"""
Plant Disease Detection & Advisory System
FastAPI Backend — Main Application Entry Point (Complete)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import structlog

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics, model_info
from app.core.rate_limiting import RateLimitMiddleware
from app.api.routes import auth, predictions, diseases, users, admin, health
from app.db.database import init_db
from app.db.mongo import close_mongo
from app.ml.models.model_manager import ModelManager

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    setup_logging()
    logger.info("Starting Plant Disease AI System", version=settings.APP_VERSION, env=settings.ENVIRONMENT)

    # Database
    await init_db()
    logger.info("PostgreSQL initialized")

    # ML Models
    model_manager = ModelManager()
    await model_manager.load_models()
    app.state.model_manager = model_manager

    # Publish model info to Prometheus
    primary = model_manager.get_primary_model()
    model_info.info({
        "name": primary.name,
        "version": primary.version,
        "device": str(primary.device),
    })

    logger.info(
        "ML models loaded",
        models=model_manager.loaded_model_names,
        device=str(primary.device),
    )

    yield

    # Graceful shutdown
    logger.info("Shutting down Plant Disease AI System")
    await model_manager.unload_models()
    await close_mongo()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Plant Disease Detection & Advisory System",
    description=(
        "AI-powered plant disease detection from leaf images. "
        "Provides disease classification, confidence scores, Grad-CAM++ visualizations, "
        "severity estimation, and treatment recommendations for farmers.\n\n"
        "**Model**: EfficientNet-B4 fine-tuned on PlantVillage (38 classes, ~98% accuracy)\n"
        "**Explainability**: Grad-CAM++ heatmaps showing disease regions\n"
        "**Treatments**: Chemical, organic, cultural, and biological options"
    ),
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware (order matters — first added = outermost wrapper) ───────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)
app.add_middleware(RateLimitMiddleware)

# ── Prometheus metrics endpoint ────────────────────────────────────────────────
setup_metrics(app)

# ── Static files ───────────────────────────────────────────────────────────────
import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── API Routers ────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"

app.include_router(health.router,      prefix=PREFIX,                  tags=["Health"])
app.include_router(auth.router,        prefix=f"{PREFIX}/auth",        tags=["Authentication"])
app.include_router(predictions.router, prefix=f"{PREFIX}/predictions", tags=["Predictions"])
app.include_router(diseases.router,    prefix=f"{PREFIX}/diseases",    tags=["Disease Database"])
app.include_router(users.router,       prefix=f"{PREFIX}/users",       tags=["Users"])
app.include_router(admin.router,       prefix=f"{PREFIX}/admin",       tags=["Admin"])
