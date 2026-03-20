"""
Grad-CAM Explainability Module
────────────────────────────────
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) and
Grad-CAM++ for visualizing which leaf regions the model focused on
when making a disease prediction.

References:
  - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
  - Chattopadhay et al. (2018) "Grad-CAM++: Improved Visual Explanations"
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    heatmap: np.ndarray           # H×W float32, range [0, 1]
    overlay: np.ndarray           # H×W×3 uint8 — colorized heatmap blended on original
    overlay_pil: Image.Image      # PIL version for easy S3 upload
    attention_boxes: List[Dict]   # bounding boxes of top disease regions
    coverage_ratio: float         # fraction of image flagged as disease-relevant


class GradCAM:
    """
    Standard Grad-CAM implementation.
    Supports any CNN backbone accessible via named modules.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):
        target = self._find_layer(self.target_layer_name)
        if target is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")

        self._hooks.append(
            target.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def _find_layer(self, name: str) -> Optional[torch.nn.Module]:
        """Navigate dotted attribute paths, e.g. 'features.8.0'."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def compute(
        self,
        input_tensor: torch.Tensor,      # (1, 3, H, W)
        class_idx: Optional[int] = None,  # None = predicted class
        original_image: Optional[np.ndarray] = None,  # BGR for overlay
    ) -> GradCAMResult:
        """
        Compute Grad-CAM for the given input.
        Returns heatmap normalized to [0, 1] and a colorized overlay.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)
        probs = torch.softmax(output, dim=-1)

        if class_idx is None:
            class_idx = output.argmax(dim=-1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        gradients = self._gradients    # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        # Global average pool gradients → weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)   # only positive influences

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Upsample to input resolution
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

        # Generate overlay
        if original_image is not None:
            overlay, attn_boxes, coverage = self._create_overlay(heatmap, original_image)
        else:
            # No source image — just colorize the heatmap
            overlay_8bit = (heatmap * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(overlay_8bit, cv2.COLORMAP_JET)
            attn_boxes = []
            coverage = float(heatmap.mean())

        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        return GradCAMResult(
            heatmap=heatmap,
            overlay=overlay,
            overlay_pil=overlay_pil,
            attention_boxes=attn_boxes,
            coverage_ratio=coverage,
        )

    def _create_overlay(
        self,
        heatmap: np.ndarray,
        bgr_image: np.ndarray,
        alpha: float = 0.45,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Blend heatmap over original image and extract attention bounding boxes.
        Returns (overlay_bgr, bounding_boxes, coverage_ratio).
        """
        # Resize image to match heatmap if needed
        h, w = heatmap.shape
        img = cv2.resize(bgr_image, (w, h))

        # Colorize heatmap
        heatmap_8bit = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

        # Alpha blend
        overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

        # Extract bounding boxes from high-attention regions
        binary = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        attention_boxes = []
        min_area = h * w * 0.005   # ignore tiny blobs < 0.5% of image
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            attention_boxes.append({
                "x": int(x), "y": int(y),
                "width": int(bw), "height": int(bh),
                "confidence": float(heatmap[y:y+bh, x:x+bw].mean()),
            })
            # Draw box on overlay
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 255, 0), 2)

        coverage = float((heatmap > threshold).mean())
        return overlay, attention_boxes, coverage

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — improved localization, especially for multiple disease regions.
    Uses second-order gradient information for more precise activation weighting.
    """

    def compute(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        original_image: Optional[np.ndarray] = None,
    ) -> GradCAMResult:
        self.model.eval()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=-1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self._gradients    # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        # Grad-CAM++ weighting
        grad_sq = gradients ** 2
        grad_cu = gradients ** 3
        sum_act = activations.sum(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        alpha_num = grad_sq
        alpha_den = 2 * grad_sq + sum_act * grad_cu
        alpha_den = torch.where(alpha_den == 0, torch.ones_like(alpha_den), alpha_den)
        alphas = alpha_num / alpha_den

        # Weights = sum over spatial positions of alpha * ReLU(gradient)
        weights = (alphas * F.relu(gradients)).mean(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

        if original_image is not None:
            overlay, attn_boxes, coverage = self._create_overlay(heatmap, original_image)
        else:
            overlay_8bit = (heatmap * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(overlay_8bit, cv2.COLORMAP_JET)
            attn_boxes = []
            coverage = float(heatmap.mean())

        return GradCAMResult(
            heatmap=heatmap,
            overlay=overlay,
            overlay_pil=Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
            attention_boxes=attn_boxes,
            coverage_ratio=coverage,
        )


def encode_overlay_to_bytes(overlay_pil: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL overlay image to JPEG bytes for S3 upload."""
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
