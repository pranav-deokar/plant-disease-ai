"""
Authentication Routes
──────────────────────
POST /api/v1/auth/register   — create new user account
POST /api/v1/auth/login      — issue JWT access + refresh tokens
POST /api/v1/auth/refresh    — rotate refresh token
POST /api/v1/auth/logout     — revoke refresh token (Redis blacklist)
GET  /api/v1/auth/me         — return current user profile
"""
"""
Authentication Routes
──────────────────────
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.models.database_models import User, UserRole
from app.schemas.prediction_schemas import (
    TokenResponse, UserLoginRequest, UserProfileResponse, UserRegisterRequest
)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    # 🔥 FIX: limit bcrypt input (max 72 chars)
    return pwd_context.hash(plain[:72])

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)


def validate_password(password: str):
    # 🔥 enforce min 8 chars and max 72 chars
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if len(password) > 72:
        raise HTTPException(status_code=400, detail="Password must be less than 72 characters.")


def create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_token_pair(user_id: str, role: str) -> TokenResponse:
    access = create_token(
        {"sub": user_id, "role": role, "type": "access"},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh = create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegisterRequest, db: AsyncSession = Depends(get_db)):

    # 🔥 PASSWORD VALIDATION
    validate_password(body.password)

    # Check uniqueness
    existing = await db.execute(
        select(User).where((User.email == body.email) | (User.username == body.username))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email or username already in use.")

    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        phone_number=body.phone_number,
        location=body.location,
        primary_crops=body.primary_crops or [],
        role=UserRole.FARMER,
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLoginRequest, db: AsyncSession = Depends(get_db)):

    result = await db.execute(select(User).where(User.email == body.email))
    user: Optional[User] = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    return create_token_pair(str(user.id), user.role.value)


@router.get("/me", response_model=UserProfileResponse)
async def get_me(request: Request, db: AsyncSession = Depends(get_db)):
    from app.api.dependencies import get_current_user
    user = await get_current_user(request=request, db=db)
    return user
