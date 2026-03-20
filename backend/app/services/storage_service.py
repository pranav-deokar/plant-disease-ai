"""
Storage Service
────────────────
Abstraction layer over AWS S3 / MinIO for:
  - Uploading images and Grad-CAM overlays
  - Generating pre-signed download URLs (short TTL)
  - Listing and deleting objects
  - Checking object existence for deduplication
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import aioboto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Async S3-compatible storage operations. Thread-safe."""

    def __init__(self):
        self._session = aioboto3.Session()
        self._bucket = settings.S3_BUCKET_NAME
        self._endpoint = settings.S3_ENDPOINT_URL or None
        self._region = settings.S3_REGION

    def _client_kwargs(self):
        kwargs = {
            "region_name": self._region,
        }
        if self._endpoint:
            kwargs["endpoint_url"] = self._endpoint
        return kwargs

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
        public: bool = False,
    ) -> str:
        """
        Upload bytes to S3/MinIO.
        Returns: pre-signed URL (TTL 7 days) or public URL if public=True.
        """
        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
        if public:
            extra_args["ACL"] = "public-read"

        async with self._session.client("s3", **self._client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                **extra_args,
            )

        if public:
            return self._public_url(key)
        return await self.get_presigned_url(key)

    async def get_presigned_url(
        self, key: str, expires_in: int = 604800   # 7 days
    ) -> str:
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )

    async def object_exists(self, key: str) -> bool:
        try:
            async with self._session.client("s3", **self._client_kwargs()) as s3:
                await s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    async def delete_object(self, key: str) -> bool:
        try:
            async with self._session.client("s3", **self._client_kwargs()) as s3:
                await s3.delete_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            logger.error(f"Failed to delete object {key}: {e}")
            return False

    async def delete_objects_older_than(self, prefix: str, cutoff: datetime) -> int:
        """List and delete objects under prefix older than cutoff. Returns deleted count."""
        deleted = 0
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                to_delete = [
                    {"Key": obj["Key"]}
                    for obj in objects
                    if obj["LastModified"].replace(tzinfo=None) < cutoff.replace(tzinfo=None)
                ]
                if to_delete:
                    response = await s3.delete_objects(
                        Bucket=self._bucket,
                        Delete={"Objects": to_delete, "Quiet": True},
                    )
                    deleted += len(to_delete) - len(response.get("Errors", []))
        return deleted

    def _public_url(self, key: str) -> str:
        if self._endpoint:
            return f"{self._endpoint}/{self._bucket}/{key}"
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"

    async def get_object_size(self, key: str) -> Optional[int]:
        try:
            async with self._session.client("s3", **self._client_kwargs()) as s3:
                response = await s3.head_object(Bucket=self._bucket, Key=key)
                return response["ContentLength"]
        except ClientError:
            return None
