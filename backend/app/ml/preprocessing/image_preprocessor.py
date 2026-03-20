"""
Image Preprocessing Pipeline
─────────────────────────────
Handles all image preparation before model inference:
  1. Validation (format, size, corruption check)
  2. Leaf segmentation (remove background)
  3. Quality enhancement (denoising, sharpening)
  4. Normalization (ImageNet stats)
  5. Augmentation (training only)
  6. Batch preprocessing for efficiency
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

class DummySettings:
    MAX_IMAGE_SIZE_MB = 10
    IMAGE_TARGET_SIZE = (224, 224)

settings = DummySettings()

logger = logging.getLogger(__name__)

# ImageNet statistics — used for pretrained backbone normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@dataclass
class PreprocessingResult:
    tensor: torch.Tensor           # shape: (1, 3, H, W), normalized
    original_image: np.ndarray     # BGR uint8 for Grad-CAM overlay
    resized_image: np.ndarray      # RGB uint8 at model input size
    image_hash: str                # SHA-256 of original bytes
    width: int
    height: int
    quality_score: float           # 0-1, image quality estimate
    is_leaf_detected: bool
    warnings: list[str]


class ImagePreprocessor:
    """
    Full preprocessing pipeline for inference.
    Thread-safe; create one instance and reuse.
    """

    SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
    MAX_SIZE_BYTES = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    def __init__(
        self,
        target_size: Tuple[int, int] = settings.IMAGE_TARGET_SIZE,
        device: str = "cpu",
    ):
        self.target_size = target_size
        self.device = device

        # Inference transform — deterministic
        self.inference_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Training transform — stochastic augmentations
        self.train_transform = T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.1, scale=(0.02, 0.15)),  # simulate occlusion
        ])

    # ── Public API ─────────────────────────────────────────────────────────────

    async def preprocess_for_inference(
        self,
        image_bytes: bytes,
        apply_enhancement: bool = True,
        segment_leaf: bool = True,
    ) -> PreprocessingResult:
        """
        Full pipeline: validate → enhance → segment → transform → tensor.
        Returns a PreprocessingResult ready for model.forward().
        """
        warnings = []

        # 1. Validate
        self._validate_bytes(image_bytes)
        image_hash = self._compute_hash(image_bytes)

        # 2. Decode
        pil_image, np_bgr = self._decode(image_bytes)
        h, w = np_bgr.shape[:2]

        # 3. Quality assessment
        quality_score = self._estimate_quality(np_bgr)
        if quality_score < 0.3:
            warnings.append(f"Low image quality detected (score={quality_score:.2f}). Results may be inaccurate.")

        # 4. Optional leaf segmentation (remove background)
        is_leaf = True
        if segment_leaf:
            np_bgr, is_leaf = self._segment_leaf(np_bgr)
            if not is_leaf:
                warnings.append("No clear leaf region detected. Predicting on full image.")

        # 5. Optional quality enhancement
        if apply_enhancement:
            pil_image = self._enhance(Image.fromarray(cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)))
        else:
            pil_image = Image.fromarray(cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB))

        # 6. Resize for Grad-CAM overlay reference
        resized_rgb = np.array(pil_image.resize(self.target_size, Image.BICUBIC))

        # 7. Apply inference transform → tensor
        tensor = self.inference_transform(pil_image).unsqueeze(0).to(self.device)

        return PreprocessingResult(
            tensor=tensor,
            original_image=np_bgr,
            resized_image=resized_rgb,
            image_hash=image_hash,
            width=w,
            height=h,
            quality_score=quality_score,
            is_leaf_detected=is_leaf,
            warnings=warnings,
        )

    def preprocess_for_training(self, pil_image: Image.Image) -> torch.Tensor:
        """Apply augmented transforms during training."""
        return self.train_transform(pil_image)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_bytes(self, data: bytes) -> None:
        if len(data) > self.MAX_SIZE_BYTES:
            raise ValueError(
                f"Image exceeds maximum size of {settings.MAX_IMAGE_SIZE_MB} MB "
                f"(got {len(data) / 1024 / 1024:.1f} MB)"
            )
        try:
            Image.open(io.BytesIO(data)).verify()
        except UnidentifiedImageError:
            raise ValueError("File is not a valid image or is corrupted.")
        except Exception as e:
            raise ValueError(f"Image validation failed: {e}")

    def _decode(self, data: bytes) -> Tuple[Image.Image, np.ndarray]:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        np_rgb = np.array(pil)
        np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        return pil, np_bgr

    def _compute_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _estimate_quality(self, bgr: np.ndarray) -> float:
        """
        Heuristic quality score combining:
          - Laplacian variance (sharpness)
          - Brightness histogram spread (exposure)
          - Color saturation (leaf greenness)
        Returns float in [0, 1].
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Sharpness (Laplacian variance, normalized)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(lap_var / 1000.0, 1.0)

        # Exposure (avoid over/under exposed)
        mean_brightness = gray.mean() / 255.0
        exposure = 1.0 - abs(mean_brightness - 0.5) * 2  # peaks at 0.5

        # Greenness (HSV saturation in green hue range)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = green_mask.sum() / (255 * bgr.shape[0] * bgr.shape[1])
        greenness = min(green_ratio * 3, 1.0)   # scale up; leaves are usually green

        return float(0.4 * sharpness + 0.3 * exposure + 0.3 * greenness)

    def _segment_leaf(self, bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        GrabCut-based leaf segmentation.
        Returns (segmented image, leaf_detected flag).
        If segmentation fails or no clear leaf, returns the original image.
        """
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Seed rectangle — assume leaf occupies 80% of center
        margin_y = int(h * 0.1)
        margin_x = int(w * 0.1)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        try:
            cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            # Definite foreground (2) and probable foreground (3) are considered leaf
            leaf_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

            # Check if enough leaf pixels were found
            leaf_ratio = leaf_mask.mean()
            if leaf_ratio < 0.1:   # less than 10% of image is leaf — segmentation unreliable
                return bgr, False

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

            # Apply mask — set background to neutral gray to avoid distracting the model
            segmented = bgr.copy()
            bg_color = np.array([200, 200, 200], dtype=np.uint8)
            segmented[leaf_mask == 0] = bg_color

            return segmented, True

        except cv2.error as e:
            logger.warning(f"GrabCut segmentation failed: {e}. Using original image.")
            return bgr, False

    def _enhance(self, pil_image: Image.Image) -> Image.Image:
        """
        Mild adaptive enhancement:
          - Sharpness: slight boost
          - Contrast: CLAHE-equivalent via PIL
          - Color: slight saturation increase to emphasize disease spots
        Deliberately mild — avoid introducing artifacts.
        """
        # Convert to numpy for CLAHE (better than PIL for adaptive contrast)
        np_img = np.array(pil_image)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_channel, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)

        # Slight sharpening
        enhanced_pil = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        # Slight color saturation boost
        enhanced_pil = ImageEnhance.Color(enhanced_pil).enhance(1.15)

        return enhanced_pil


class BatchPreprocessor:
    """
    Efficient batch preprocessing for model retraining and dataset preparation.
    Uses multiprocessing for CPU-bound image ops.
    """

    def __init__(self, preprocessor: Optional[ImagePreprocessor] = None):
        self.preprocessor = preprocessor or ImagePreprocessor()

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        class_mapping: dict,
        num_workers: int = 4,
    ) -> dict:
        """
        Processes an entire PlantVillage-style directory tree:
          input_dir/
            Tomato___Early_blight/  → class label
              img001.jpg
              ...
        Saves processed tensors and returns class statistics.
        """
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        stats = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = []

        for class_dir in sorted(input_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_idx = class_mapping.get(class_name)
            if class_idx is None:
                logger.warning(f"Unknown class directory: {class_name}")
                continue

            out_class_dir = output_dir / class_name
            out_class_dir.mkdir(exist_ok=True)
            stats[class_name] = {"total": 0, "processed": 0, "errors": 0}

            for img_path in class_dir.glob("**/*"):
                if img_path.suffix.lower().lstrip(".") in ImagePreprocessor.SUPPORTED_FORMATS:
                    tasks.append((img_path, out_class_dir, class_idx, class_name))
                    stats[class_name]["total"] += 1

        def _process_one(args):
            img_path, out_dir, class_idx, class_name = args
            try:
                img = Image.open(img_path).convert("RGB")
                # Save resized version for fast DataLoader reads
                resized = img.resize((456, 456), Image.BICUBIC)   # slightly larger than model input
                resized.save(out_dir / img_path.name, quality=90, optimize=True)
                return class_name, True
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                return class_name, False

        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            for class_name, success in tqdm(exe.map(_process_one, tasks), total=len(tasks)):
                if success:
                    stats[class_name]["processed"] += 1
                else:
                    stats[class_name]["errors"] += 1

        return stats
