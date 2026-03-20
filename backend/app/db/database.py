"""
Async SQLAlchemy database configuration.
Uses asyncpg driver for non-blocking PostgreSQL access.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
import logging
logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,         # detect stale connections
    pool_recycle=3600,          # recycle every hour
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,     # prevent lazy-load after commit in async context
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables that don't exist yet. Run once at startup."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning(f"DB init warning: {e}. Tables may already exist.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — provides a DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
