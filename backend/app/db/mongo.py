"""
Async MongoDB client (Motor) for the disease knowledge base.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

_client: AsyncIOMotorClient | None = None


async def get_mongo_db() -> AsyncIOMotorDatabase:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=20,
        )
    return _client[settings.MONGODB_DB_NAME]


async def close_mongo():
    global _client
    if _client is not None:
        _client.close()
        _client = None
