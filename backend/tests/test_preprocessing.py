"""
Unit Tests — Image Preprocessing Pipeline
───────────────────────────────────────────
Tests each stage of the preprocessing pipeline in isolation.
All tests run without GPU or network access.
"""

import asyncio
import io
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    BatchPreprocessor,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from app.utils.image_utils import (
    detect_image_format,
    generate_thumbnail,
    image_to_base64,
    base64_to_image,
    compute_green_index,
    add_prediction_watermark,
    resize_to_max_dimension,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_rgb_image(width=256, height=256, color=(100, 200, 80)) -> bytes:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def make_png_image(width=128, height=128) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def make_webp_image() -> bytes:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 1] = 180
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=80)
    return buf.getvalue()


# ── ImagePreprocessor tests ────────────────────────────────────────────────────

class TestImagePreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor(target_size=(224, 224))

    @pytest.fixture
    def green_leaf_bytes(self):
        return make_rgb_image(color=(30, 170, 40))

    # ── Output shape and dtype ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tensor_shape(self, preprocessor, green_leaf_bytes):
        result = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        assert result.tensor.shape == (1, 3, 224, 224)
        assert result.tensor.dtype == torch.float32

    @pytest.mark.asyncio
    async def test_different_target_size(self, green_leaf_bytes):
        p = ImagePreprocessor(target_size=(380, 380))
        result = await p.preprocess_for_inference(green_leaf_bytes)
        assert result.tensor.shape == (1, 3, 380, 380)

    @pytest.mark.asyncio
    async def test_normalization_range(self, preprocessor, green_leaf_bytes):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        result = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        t = result.tensor
        assert t.min().item() > -4.0, "Normalized tensor too negative"
        assert t.max().item() < 4.0,  "Normalized tensor too positive"

    # ── Hash ──────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_image_hash_sha256(self, preprocessor, green_leaf_bytes):
        result = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        assert len(result.image_hash) == 64           # SHA-256 hex string
        assert all(c in "0123456789abcdef" for c in result.image_hash)

    @pytest.mark.asyncio
    async def test_identical_images_same_hash(self, preprocessor, green_leaf_bytes):
        r1 = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        r2 = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        assert r1.image_hash == r2.image_hash

    @pytest.mark.asyncio
    async def test_different_images_different_hash(self, preprocessor):
        r1 = await preprocessor.preprocess_for_inference(make_rgb_image(color=(100, 200, 80)))
        r2 = await preprocessor.preprocess_for_inference(make_rgb_image(color=(200, 100, 50)))
        assert r1.image_hash != r2.image_hash

    # ── Format support ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_accepts_png(self, preprocessor):
        png_bytes = make_png_image()
        result = await preprocessor.preprocess_for_inference(png_bytes)
        assert result.tensor.shape[0] == 1

    @pytest.mark.asyncio
    async def test_accepts_webp(self, preprocessor):
        webp_bytes = make_webp_image()
        result = await preprocessor.preprocess_for_inference(webp_bytes)
        assert result.tensor.shape[0] == 1

    # ── Validation ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_rejects_empty_bytes(self, preprocessor):
        with pytest.raises(ValueError, match="not a valid image|corrupted"):
            await preprocessor.preprocess_for_inference(b"")

    @pytest.mark.asyncio
    async def test_rejects_random_bytes(self, preprocessor):
        with pytest.raises(ValueError):
            await preprocessor.preprocess_for_inference(b"not-an-image-at-all-xyz")

    @pytest.mark.asyncio
    async def test_rejects_oversized_file(self, preprocessor):
        oversized = b"A" * (11 * 1024 * 1024)  # 11 MB
        with pytest.raises(ValueError, match="exceeds maximum size"):
            await preprocessor.preprocess_for_inference(oversized)

    @pytest.mark.asyncio
    async def test_accepts_large_valid_image(self, preprocessor):
        """9 MB valid image should be accepted."""
        large_arr = np.zeros((3000, 3000, 3), dtype=np.uint8)
        large_arr[:, :, 1] = 150
        pil = Image.fromarray(large_arr, "RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        large_bytes = buf.getvalue()
        if len(large_bytes) < 9 * 1024 * 1024:
            result = await preprocessor.preprocess_for_inference(large_bytes)
            assert result.tensor.shape == (1, 3, 224, 224)

    # ── Quality score ─────────────────────────────────────────────────────────

    def test_quality_score_sharp_image(self, preprocessor):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create high-contrast checkerboard (sharp)
        arr[::8, :, :] = 255
        arr[:, ::8, :] = 255
        arr[:, :, 1] = 150
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        score = preprocessor._estimate_quality(bgr)
        assert 0.0 <= score <= 1.0

    def test_quality_score_solid_color_image(self, preprocessor):
        """Solid color = no edges = low sharpness."""
        arr = np.full((256, 256, 3), 128, dtype=np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        score = preprocessor._estimate_quality(bgr)
        # Solid image: low sharpness but might have decent brightness/greenness
        assert 0.0 <= score <= 1.0

    def test_quality_score_green_image_higher_than_gray(self, preprocessor):
        green_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        green_arr[:, :, 1] = 180
        gray_arr = np.full((256, 256, 3), 128, dtype=np.uint8)

        green_bgr = cv2.cvtColor(green_arr, cv2.COLOR_RGB2BGR)
        gray_bgr  = cv2.cvtColor(gray_arr,  cv2.COLOR_RGB2BGR)

        green_score = preprocessor._estimate_quality(green_bgr)
        gray_score  = preprocessor._estimate_quality(gray_bgr)
        assert green_score > gray_score, "Green image should score higher than gray"

    # ── Enhancement ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_enhancement_does_not_change_tensor_shape(self, preprocessor, green_leaf_bytes):
        r_enhanced    = await preprocessor.preprocess_for_inference(green_leaf_bytes, apply_enhancement=True)
        r_noenhanced  = await preprocessor.preprocess_for_inference(green_leaf_bytes, apply_enhancement=False)
        assert r_enhanced.tensor.shape == r_noenhanced.tensor.shape

    # ── Training transform ────────────────────────────────────────────────────

    def test_train_transform_output_shape(self, preprocessor, green_leaf_bytes):
        pil = Image.open(io.BytesIO(green_leaf_bytes))
        tensor = preprocessor.preprocess_for_training(pil)
        assert tensor.shape == (3, 224, 224)

    def test_train_transform_stochastic(self, preprocessor, green_leaf_bytes):
        """Two applications of train transform should produce different results."""
        pil = Image.open(io.BytesIO(green_leaf_bytes))
        t1 = preprocessor.preprocess_for_training(pil)
        t2 = preprocessor.preprocess_for_training(pil)
        # With high probability the random augmentations differ
        # (horizontal flip alone gives 50% chance of difference)
        assert not torch.allclose(t1, t2), "Train transforms should be stochastic"

    # ── Original image preservation ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_original_image_is_bgr_numpy(self, preprocessor, green_leaf_bytes):
        result = await preprocessor.preprocess_for_inference(green_leaf_bytes)
        assert isinstance(result.original_image, np.ndarray)
        assert result.original_image.dtype == np.uint8
        assert result.original_image.ndim == 3
        assert result.original_image.shape[2] == 3   # BGR

    @pytest.mark.asyncio
    async def test_image_dimensions_recorded(self, preprocessor):
        wide_image = make_rgb_image(width=640, height=480)
        result = await preprocessor.preprocess_for_inference(wide_image)
        assert result.width == 640
        assert result.height == 480


# ── Image utility tests ───────────────────────────────────────────────────────

class TestImageUtils:

    def test_detect_jpeg_format(self):
        jpeg_bytes = make_rgb_image()
        fmt = detect_image_format(jpeg_bytes)
        assert fmt == "jpeg"

    def test_detect_png_format(self):
        png_bytes = make_png_image()
        fmt = detect_image_format(png_bytes)
        assert fmt == "png"

    def test_detect_webp_format(self):
        webp_bytes = make_webp_image()
        fmt = detect_image_format(webp_bytes)
        assert fmt == "webp"

    def test_detect_unknown_format(self):
        fmt = detect_image_format(b"NOTANIMAGE_xyz123")
        assert fmt is None

    def test_thumbnail_size(self):
        large = make_rgb_image(width=1920, height=1080)
        thumb = generate_thumbnail(large, size=(256, 256))
        pil = Image.open(io.BytesIO(thumb))
        assert pil.width <= 256
        assert pil.height <= 256

    def test_thumbnail_smaller_than_original(self):
        original = make_rgb_image(width=512, height=512)
        thumb = generate_thumbnail(original, size=(128, 128))
        assert len(thumb) < len(original)

    def test_base64_roundtrip(self):
        original = make_rgb_image()
        encoded = image_to_base64(original)
        assert encoded.startswith("data:image/jpeg;base64,")
        decoded = base64_to_image(encoded)
        assert decoded == original

    def test_green_index_for_green_image(self):
        green = make_rgb_image(color=(30, 200, 40))
        score = compute_green_index(green)
        assert score > 0.2, f"Green image should have high green index, got {score}"

    def test_green_index_for_red_image(self):
        red = make_rgb_image(color=(220, 30, 30))
        score = compute_green_index(red)
        assert score < 0.3, f"Red image should have low green index, got {score}"

    def test_green_index_range(self):
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]:
            img = make_rgb_image(color=color)
            score = compute_green_index(img)
            assert 0.0 <= score <= 1.0, f"Green index out of range for color {color}: {score}"

    def test_resize_to_max_dimension_large_image(self):
        large = make_rgb_image(width=4000, height=3000)
        resized = resize_to_max_dimension(large, max_dim=1920)
        pil = Image.open(io.BytesIO(resized))
        assert max(pil.size) <= 1920

    def test_resize_to_max_dimension_small_image_unchanged(self):
        small = make_rgb_image(width=640, height=480)
        result = resize_to_max_dimension(small, max_dim=1920)
        pil_orig   = Image.open(io.BytesIO(small))
        pil_result = Image.open(io.BytesIO(result))
        assert pil_result.size == pil_orig.size

    def test_watermark_returns_jpeg(self):
        img = make_rgb_image()
        watermarked = add_prediction_watermark(
            img,
            disease_name="Tomato — Early Blight",
            confidence=0.95,
            severity="moderate",
        )
        assert watermarked[:3] == b"\xff\xd8\xff"   # JPEG magic bytes

    def test_watermark_does_not_enlarge_dimensions(self):
        img = make_rgb_image(width=512, height=512)
        watermarked = add_prediction_watermark(img, "Test Disease", 0.9, "mild")
        pil = Image.open(io.BytesIO(watermarked))
        assert pil.width == 512
        assert pil.height == 512
