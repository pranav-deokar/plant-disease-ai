"""
Health Check Route
───────────────────
GET /api/v1/health — liveness probe (returns 200 if API is running)
GET /api/v1/health/ready — readiness probe (checks DB, Redis, models)
"""

import time
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.core.config import settings

router = APIRouter()
_start_time = time.time()


@router.get("/health", tags=["Health"])
async def liveness():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": int(time.time() - _start_time),
    }


@router.get("/health/ready", tags=["Health"])
async def readiness(request: Request, db: AsyncSession = Depends(get_db)):
    checks = {}

    # DB check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Model check
    try:
        model_manager = request.app.state.model_manager
        checks["models"] = model_manager.loaded_model_names
    except Exception as e:
        checks["models"] = f"error: {e}"

    # Redis check
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" or isinstance(v, list) for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
