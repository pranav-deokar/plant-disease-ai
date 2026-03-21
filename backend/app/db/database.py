"""
Async SQLAlchemy database configuration.
Fixed for Supabase + PgBouncer (no prepared statements).
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# 🔥 FIX: disable prepared statements (PgBouncer compatible)
engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args={
        "statement_cache_size": 0,   # disable asyncpg cache
        "prepared_statement_cache_size": 0  # 🔥 THIS IS MISSING
    },
    poolclass=None,  # 🔥 disable SQLAlchemy pooling (important for PgBouncer)
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables that don't exist yet."""
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
