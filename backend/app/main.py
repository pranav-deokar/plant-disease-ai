"""
Plant Disease Detection & Advisory System
FastAPI Backend - Main Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import structlog
import os
import asyncio

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import auth, predictions, diseases, users, admin, health
from app.db.database import init_db
from app.ml.models.model_manager import ModelManager

logger = structlog.get_logger()


@asynccontextmanager

async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Plant Disease AI System", version=settings.APP_VERSION)

    # Initialize DB (safe)
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Model manager (safe)
    model_manager = ModelManager()
    app.state.model_manager = model_manager

    import asyncio
    asyncio.create_task(model_manager.load_models())

    logger.info("Startup completed")

    yield

    logger.info("Shutting down Plant Disease AI System")
    try:
        await model_manager.unload_models()
    except:
        pass

app = FastAPI(
    title="Plant Disease Detection & Advisory System",
    description=(
        "AI-powered plant disease detection from leaf images. "
        "Provides disease classification, confidence scores, Grad-CAM visualizations, "
        "and treatment recommendations for farmers."
    ),
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ───────────────────────────────────────────────────────────────
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(diseases.router, prefix="/api/v1/diseases", tags=["Disease Database"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
