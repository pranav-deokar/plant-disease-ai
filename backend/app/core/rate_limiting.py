"""
Rate Limiting Middleware
─────────────────────────
Sliding-window rate limiter backed by Redis.
Configurable per-route with different limits:
  - General API: 60 req/min
  - Prediction endpoint: 10 req/min (CPU/GPU protection)
  - Auth endpoints: 20 req/min (brute-force protection)

Limits are applied per IP address. Authenticated users get
their user_id as the key instead of IP for higher limits.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as aioredis

from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.
    Uses Redis ZSET with score = timestamp for O(log N) operations.
    """

    # (path_prefix, requests_per_minute)
    ROUTE_LIMITS: list[tuple[str, int]] = [
        ("/api/v1/predictions", settings.RATE_LIMIT_PER_MINUTE // 6),   # 10/min
        ("/api/v1/auth/login",  20),
        ("/api/v1/auth/register", 5),
        ("/api/v1/",            settings.RATE_LIMIT_PER_MINUTE),         # 60/min default
    ]

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _get_limit(self, path: str) -> int:
        for prefix, limit in self.ROUTE_LIMITS:
            if path.startswith(prefix):
                return limit
        return settings.RATE_LIMIT_PER_MINUTE

    def _get_client_key(self, request: Request) -> str:
        # Prefer authenticated user ID over IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"ratelimit:user:{user_id}"
        ip = request.client.host if request.client else "unknown"
        return f"ratelimit:ip:{ip}"

    async def dispatch(self, request: Request, call_next):
        # Skip health checks and metrics
        if request.url.path in {"/api/v1/health", "/metrics", "/api/v1/health/ready"}:
            return await call_next(request)

        try:
            redis = await self._get_redis()
            key = self._get_client_key(request)
            limit = self._get_limit(request.url.path)
            window = 60   # seconds

            now = time.time()
            window_start = now - window

            pipe = redis.pipeline()
            # Remove old entries outside window
            pipe.zremrangebyscore(key, 0, window_start)
            # Count requests in window
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Set TTL
            pipe.expire(key, window + 10)

            results = await pipe.execute()
            request_count = results[1]

            # Remaining = limit - count (before adding current)
            remaining = max(0, limit - request_count - 1)
            reset_at = int(now + window)

            # Add rate limit headers to response
            if request_count >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded. Maximum {limit} requests per minute.",
                        "retry_after": window,
                    },
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_at),
                        "Retry-After": str(window),
                    },
                )

            response: Response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_at)
            return response

        except Exception:
            # Redis unavailable — fail open (don't block requests)
            return await call_next(request)
