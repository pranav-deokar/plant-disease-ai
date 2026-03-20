"""
Image Utility Helpers
──────────────────────
Shared utilities for image handling across the application:
  - Format detection
  - Safe thumbnail generation
  - Base64 encoding/decoding
  - EXIF data extraction
  - Watermarking for disease detection overlays
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags

logger = logging.getLogger(__name__)

# Maximum dimensions for thumbnails stored alongside predictions
THUMBNAIL_SIZE = (256, 256)


def detect_image_format(data: bytes) -> Optional[str]:
    """
    Detect image format from magic bytes — more reliable than file extension.
    Returns lowercase format string or None.
    """
    signatures = {
        b"\xff\xd8\xff": "jpeg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"RIFF": "webp",   # followed by WEBP at offset 8
        b"BM": "bmp",
        b"\x49\x49\x2a\x00": "tiff",  # little-endian TIFF
        b"\x4d\x4d\x00\x2a": "tiff",  # big-endian TIFF
    }
    for magic, fmt in signatures.items():
        if data[:len(magic)] == magic:
            if fmt == "webp" and data[8:12] != b"WEBP":
                continue
            return fmt
    return None


def generate_thumbnail(
    image_bytes: bytes,
    size: Tuple[int, int] = THUMBNAIL_SIZE,
    quality: int = 75,
) -> bytes:
    """
    Generate a JPEG thumbnail from image bytes.
    Preserves aspect ratio by center-cropping to fill the target size.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to fit inside target while preserving aspect ratio
        pil.thumbnail(size, Image.LANCZOS)

        # Center-crop to exact size
        w, h = pil.size
        left   = (w - min(w, size[0])) // 2
        top    = (h - min(h, size[1])) // 2
        right  = left + min(w, size[0])
        bottom = top  + min(h, size[1])
        pil = pil.crop((left, top, right, bottom))

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")
        return image_bytes


def image_to_base64(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode image bytes to a data URI for inline embedding."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def base64_to_image(data_uri: str) -> bytes:
    """Decode a data URI back to raw bytes."""
    if "," in data_uri:
        _, b64_part = data_uri.split(",", 1)
    else:
        b64_part = data_uri
    return base64.b64decode(b64_part)


def extract_exif(image_bytes: bytes) -> dict:
    """
    Extract EXIF metadata from JPEG images.
    Returns a flat dict with human-readable keys.
    Silently returns {} for images without EXIF.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        raw_exif = pil.getexif()
        if not raw_exif:
            return {}

        result = {}
        for tag_id, value in raw_exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Skip binary blobs and MakerNote
            if isinstance(value, bytes) or tag in {"MakerNote", "UserComment"}:
                continue
            result[tag] = str(value)

        # GPS coordinates
        gps_info = raw_exif.get_ifd(0x8825)   # GPSInfo IFD
        if gps_info:
            lat = _parse_gps_coord(gps_info.get(2), gps_info.get(1))
            lon = _parse_gps_coord(gps_info.get(4), gps_info.get(3))
            if lat and lon:
                result["gps_latitude"]  = lat
                result["gps_longitude"] = lon

        return result
    except Exception:
        return {}


def _parse_gps_coord(coord_tuple, ref) -> Optional[float]:
    """Convert EXIF GPS rational tuples to decimal degrees."""
    try:
        degrees   = coord_tuple[0]
        minutes   = coord_tuple[1]
        seconds   = coord_tuple[2]
        decimal   = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def add_prediction_watermark(
    image_bytes: bytes,
    disease_name: str,
    confidence: float,
    severity: str,
) -> bytes:
    """
    Add a semi-transparent prediction label overlay to the image.
    Useful for generating shareable result images.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil.size
        draw = ImageDraw.Draw(pil, "RGBA")

        # Bottom banner
        banner_h = max(60, h // 8)
        draw.rectangle(
            [(0, h - banner_h), (w, h)],
            fill=(0, 0, 0, 160),
        )

        # Text
        severity_colors = {
            "healthy":  (46, 204, 113),
            "mild":     (241, 196, 15),
            "moderate": (230, 126, 34),
            "severe":   (231, 76, 60),
            "critical": (192, 57, 43),
        }
        sev_color = severity_colors.get(severity, (255, 255, 255))

        font_size = max(14, banner_h // 3)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 4)
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        draw.text(
            (10, h - banner_h + 8),
            disease_name,
            font=font,
            fill=(255, 255, 255, 240),
        )
        draw.text(
            (10, h - banner_h + 8 + font_size + 4),
            f"Confidence: {confidence:.1%}  |  Severity: {severity.upper()}",
            font=small_font,
            fill=(*sev_color, 220),
        )

        # Logo watermark top-right
        draw.text(
            (w - 140, 10),
            "🌿 Plant Disease AI",
            font=small_font,
            fill=(255, 255, 255, 180),
        )

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Watermark failed: {e}")
        return image_bytes


def resize_to_max_dimension(
    image_bytes: bytes,
    max_dim: int = 1920,
    quality: int = 88,
) -> bytes:
    """
    Downsample very large images before upload to stay under size limits.
    Preserves aspect ratio; does nothing if image is already within bounds.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil.size
        if max(w, h) <= max_dim:
            return image_bytes

        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return image_bytes


def compute_green_index(image_bytes: bytes) -> float:
    """
    Simple greenness index based on excess green (ExG = 2G - R - B).
    Used as a quick heuristic for leaf detection.
    Returns float in [0, 1]; higher = more green.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
        arr = np.array(pil, dtype=float)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        exg = (2 * g - r - b) / 255.0
        return float(np.clip(exg.mean(), 0, 1))
    except Exception:
        return 0.0
