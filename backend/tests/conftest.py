"""
Pytest Configuration & Shared Fixtures
────────────────────────────────────────
Provides:
  - Async test session setup
  - In-memory SQLite database for isolation
  - HTTP test client with overridden dependencies
  - Factory fixtures for common model instances
  - Sample image bytes fixture
"""

import asyncio
import io
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base, get_db
from app.main import app
from app.models.database_models import User, UserRole
from app.core.security import hash_password
from app.ml.models.model_manager import DISEASE_CLASSES, NUM_CLASSES
import torch


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Shared event loop across all tests in the session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── In-memory test database ───────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provides a transactional test DB session that rolls back after each test."""
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ── App overrides ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with database override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Inject mock model manager
    app.state.model_manager = _make_mock_model_manager()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()


def _make_mock_model_manager():
    """Lightweight mock model manager returning deterministic predictions."""
    mm = MagicMock()

    probs = torch.zeros(1, NUM_CLASSES)
    probs[0, 29] = 0.942   # tomato___early_blight
    probs[0, 36] = 0.041
    probs[0, 34] = 0.017

    loaded = MagicMock()
    loaded.model = MagicMock(return_value=probs)
    loaded.name = "efficientnet_b4_plantvillage"
    loaded.version = "v1.0.0-test"
    loaded.device = torch.device("cpu")

    mm.get_primary_model.return_value = loaded
    mm.loaded_model_names = ["efficientnet_b4_plantvillage"]
    mm.record_inference = MagicMock()
    return mm


# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def green_leaf_bytes() -> bytes:
    """Synthetic 256×256 green leaf JPEG."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 1] = 160   # green
    arr[80:140, 60:180] = [30, 90, 20]   # darker patch (simulated disease)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def white_noise_bytes() -> bytes:
    """Random noise image — should get a low quality score."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── Model fixtures ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a standard farmer user."""
    user = User(
        id=uuid.uuid4(),
        email=f"farmer_{uuid.uuid4().hex[:8]}@test.com",
        username=f"farmer_{uuid.uuid4().hex[:8]}",
        hashed_password=hash_password("testpassword123"),
        full_name="Test Farmer",
        role=UserRole.FARMER,
        is_active=True,
        is_verified=True,
        primary_crops=["tomato", "potato"],
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create an admin user."""
    user = User(
        id=uuid.uuid4(),
        email=f"admin_{uuid.uuid4().hex[:8]}@test.com",
        username=f"admin_{uuid.uuid4().hex[:8]}",
        hashed_password=hash_password("adminpassword123"),
        full_name="Test Admin",
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient, test_user: User) -> dict:
    """Return Authorization headers for test_user."""
    response = await client.post("/api/v1/auth/login", json={
        "email": test_user.email,
        "password": "testpassword123",
    })
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def admin_headers(client: AsyncClient, admin_user: User) -> dict:
    """Return Authorization headers for admin_user."""
    response = await client.post("/api/v1/auth/login", json={
        "email": admin_user.email,
        "password": "adminpassword123",
    })
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
