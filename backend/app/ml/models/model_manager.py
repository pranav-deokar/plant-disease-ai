"""
ML Model Definitions and Model Manager
────────────────────────────────────────
- EfficientNetB4PlantDisease: primary production model (highest accuracy)
- MobileNetV3PlantDisease: lightweight fallback (mobile / edge deployment)
- ModelManager: singleton that loads, serves, and hot-swaps models
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Class Definitions ──────────────────────────────────────────────────────────
# Must match the PlantVillage training labels exactly (38 classes)
DISEASE_CLASSES: List[str] = [
    "apple___apple_scab",
    "apple___black_rot",
    "apple___cedar_apple_rust",
    "apple___healthy",
    "blueberry___healthy",
    "cherry___healthy",
    "cherry___powdery_mildew",
    "corn___cercospora_leaf_spot",
    "corn___common_rust",
    "corn___healthy",
    "corn___northern_leaf_blight",
    "grape___black_rot",
    "grape___esca_black_measles",
    "grape___healthy",
    "grape___leaf_blight",
    "orange___haunglongbing",
    "peach___bacterial_spot",
    "peach___healthy",
    "pepper___bacterial_spot",
    "pepper___healthy",
    "potato___early_blight",
    "potato___healthy",
    "potato___late_blight",
    "raspberry___healthy",
    "soybean___healthy",
    "squash___powdery_mildew",
    "strawberry___healthy",
    "strawberry___leaf_scorch",
    "tomato___bacterial_spot",
    "tomato___early_blight",
    "tomato___healthy",
    "tomato___late_blight",
    "tomato___leaf_mold",
    "tomato___mosaic_virus",
    "tomato___septoria_leaf_spot",
    "tomato___spider_mites",
    "tomato___target_spot",
    "tomato___yellow_leaf_curl_virus",
]

NUM_CLASSES = len(DISEASE_CLASSES)


# ── Model Architectures ────────────────────────────────────────────────────────

class EfficientNetB4PlantDisease(nn.Module):
    """
    EfficientNet-B4 fine-tuned for PlantVillage disease classification.

    Architecture choices:
    - Unfreeze last 3 blocks for fine-tuning (blocks 5-7)
    - Custom classifier head with dropout for regularization
    - Input: 380×380 (EfficientNet-B4 native resolution)
    - Output: 38 logits → softmax probabilities
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.efficientnet_b4(weights=weights)

        # Freeze early layers — keep low-level feature extractors from ImageNet
        for name, param in backbone.named_parameters():
            param.requires_grad = False

        # Unfreeze the last 3 MBConv blocks + final conv
        for name, param in backbone.named_parameters():
            if any(f"features.{i}" in name for i in [5, 6, 7, 8]):
                param.requires_grad = True

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # EfficientNet-B4 feature dim before head = 1792
        feature_dim = 1792

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),                      # EfficientNet native activation
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class MobileNetV3PlantDisease(nn.Module):
    """
    MobileNetV3-Large fine-tuned for plant disease classification.
    Lighter model for edge deployment / low-latency requirements.
    Input: 224×224
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.mobilenet_v3_large(weights=weights)

        # Freeze early backbone layers
        for i, block in enumerate(backbone.features):
            if i < 10:
                for p in block.parameters():
                    p.requires_grad = False

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        feature_dim = 960   # MobileNetV3-Large final feature dim

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Any] = {
    "efficientnet_b4_plantvillage": {
        "class": EfficientNetB4PlantDisease,
        "input_size": 380,
        "weight_file": "efficientnet_b4_plantvillage.pth",
    },
    "mobilenet_v3_plantvillage": {
        "class": MobileNetV3PlantDisease,
        "input_size": 224,
        "weight_file": "mobilenet_v3_plantvillage.pth",
    },
}


# ── Model Manager ──────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    name: str
    version: str
    model: nn.Module
    device: torch.device
    input_size: int
    loaded_at: float = field(default_factory=time.time)
    inference_count: int = 0
    total_inference_ms: float = 0.0

    @property
    def avg_inference_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_ms / self.inference_count


class ModelManager:
    """
    Singleton that manages model lifecycle:
      - Load model weights from disk / S3
      - Serve primary + fallback models
      - Hot-swap on retraining events
      - Track inference statistics per model
    """

    def __init__(self):
        self._models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._device = self._select_device()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS backend selected")
        else:
            device = torch.device("cpu")
            logger.info("Running on CPU — inference will be slower")
        return device

    async def load_models(self):
        """Load primary and fallback models at startup."""
        try:
            await self._load_model(settings.PRIMARY_MODEL_NAME, settings.PRIMARY_MODEL_VERSION)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ERROR LOADING MODEL:", str(e))
            raise e
        if settings.FALLBACK_MODEL_NAME:
            try:
                await self._load_model(settings.FALLBACK_MODEL_NAME, "v1.0.0")
            except Exception as e:
                logger.warning(f"Fallback model failed to load: {e}. Continuing.")
    async def _load_model(self, model_name: str, version: str):
        async with self._lock:
            if model_name in self._models:
                logger.info(f"Model {model_name} already loaded, skipping")
                return

            spec = MODEL_REGISTRY.get(model_name)
            if spec is None:
                raise ValueError(f"Unknown model: {model_name}")

            weight_path = settings.MODEL_DIR / spec["weight_file"]

            if not weight_path.exists() or weight_path.stat().st_size == 0:
                logger.warning(f"Weights not found at {weight_path}. Downloading from S3...")
                await self._download_weights(spec["weight_file"], weight_path)

            # Instantiate and load weights
            model_cls = spec["class"]
            model = model_cls(num_classes=NUM_CLASSES, pretrained=False)

            state_dict = torch.load(weight_path, map_location=self._device, weights_only=True)
            # Handle DataParallel-saved weights
            if all(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            print("MISSING KEYS:", missing)
            print("UNEXPECTED KEYS:", unexpected)
            model.to(self._device)
            model.eval()

            # Warm up JIT / kernel compilations
            dummy = torch.zeros(1, 3, spec["input_size"], spec["input_size"], device=self._device)
            with torch.no_grad():
                _ = model(dummy)

            self._models[model_name] = LoadedModel(
                name=model_name,
                version=version,
                model=model,
                device=self._device,
                input_size=spec["input_size"],
            )
            logger.info(f"Model loaded: {model_name} v{version} on {self._device}")

    async def _download_weights(self, filename: str, dest: Path):
        """Download model weights from S3 if not cached locally."""
        import aioboto3
        dest.parent.mkdir(parents=True, exist_ok=True)
        session = aioboto3.Session()
        async with session.client(
            "s3",
            region_name=settings.S3_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL or None,
        ) as s3:
            await s3.download_file(settings.S3_BUCKET_NAME, f"models/{filename}", str(dest))
        logger.info(f"Downloaded model weights: {filename}")

    def get_primary_model(self) -> LoadedModel:
        model = self._models.get(settings.PRIMARY_MODEL_NAME)
        if model is None:
            raise RuntimeError("Primary model not loaded")
        return model

    def get_fallback_model(self) -> Optional[LoadedModel]:
        return self._models.get(settings.FALLBACK_MODEL_NAME)

    @property
    def loaded_model_names(self) -> List[str]:
        return list(self._models.keys())

    def record_inference(self, model_name: str, duration_ms: float):
        if m := self._models.get(model_name):
            m.inference_count += 1
            m.total_inference_ms += duration_ms

    async def swap_primary_model(self, new_model_name: str, new_version: str):
        """Hot-swap primary model after retraining — zero downtime."""
        await self._load_model(new_model_name, new_version)
        async with self._lock:
            old_primary = settings.PRIMARY_MODEL_NAME
            settings.PRIMARY_MODEL_NAME = new_model_name
            logger.info(f"Primary model swapped: {old_primary} → {new_model_name}")

    async def unload_models(self):
        async with self._lock:
            for name, loaded in self._models.items():
                del loaded.model
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.info(f"Model unloaded: {name}")
            self._models.clear()
