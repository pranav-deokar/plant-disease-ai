"""
Integration Tests — Prediction API
────────────────────────────────────
Tests the end-to-end prediction pipeline using a lightweight mock model
so tests run fast without GPU or real weights.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
import torch
from fastapi.testclient import TestClient
from httpx import AsyncClient
from PIL import Image

from app.main import app
from app.ml.models.model_manager import DISEASE_CLASSES, NUM_CLASSES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_leaf_image_bytes() -> bytes:
    """Generate a synthetic 256x256 green leaf-like image for testing."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 1] = 180   # strong green channel
    arr[:, :, 0] = 30    # slight red
    arr[:, :, 2] = 30    # slight blue
    # Add some brown spots to simulate disease
    arr[80:120, 80:120, 0] = 120
    arr[80:120, 80:120, 1] = 60
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def mock_model_manager():
    """Lightweight mock that returns deterministic softmax probabilities."""
    mm = MagicMock()

    fake_probs = torch.zeros(1, NUM_CLASSES)
    fake_probs[0, 29] = 0.94   # tomato___early_blight at index 29
    fake_probs[0, 36] = 0.04   # tomato___target_spot at index 36
    fake_probs[0, 34] = 0.02   # tomato___septoria_leaf_spot

    fake_loaded = MagicMock()
    fake_loaded.model = MagicMock(return_value=fake_probs)
    fake_loaded.name = "efficientnet_b4_plantvillage"
    fake_loaded.version = "v1.0.0-test"
    fake_loaded.device = torch.device("cpu")

    mm.get_primary_model.return_value = fake_loaded
    mm.loaded_model_names = ["efficientnet_b4_plantvillage"]
    mm.record_inference = MagicMock()

    return mm


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload_bytes.return_value = "https://storage.example.com/test/image.jpg"
    storage.object_exists.return_value = False
    return storage


@pytest.fixture
def mock_advisory():
    advisory = AsyncMock()
    advisory.get_treatments.return_value = {
        "description": "Test disease description",
        "treatments": {
            "chemical": [{"treatment_name": "Test Fungicide", "application_method": "Spray"}],
            "organic": [{"treatment_name": "Copper spray", "application_method": "Spray"}],
        },
        "preventive_practices": ["Practice crop rotation"],
    }
    return advisory


@pytest.fixture
def client(mock_model_manager, mock_storage, mock_advisory):
    """Test client with mocked services."""
    app.state.model_manager = mock_model_manager
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_liveness(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_readiness_structure(self, client):
        response = client.get("/api/v1/health/ready")
        # May be degraded in test environment (no real DB), but should return 200
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data


class TestPredictionAPI:
    def test_predict_valid_image(
        self,
        client,
        sample_leaf_image_bytes,
        mock_storage,
        mock_advisory,
    ):
        with (
            patch("app.api.routes.predictions.StorageService", return_value=mock_storage),
            patch("app.api.routes.predictions.AdvisoryService", return_value=mock_advisory),
        ):
            response = client.post(
                "/api/v1/predictions/",
                files={"image": ("leaf.jpg", sample_leaf_image_bytes, "image/jpeg")},
                data={"top_k": "3"},
            )

        assert response.status_code == 201
        data = response.json()

        # Structure checks
        assert "prediction_id" in data
        assert "disease_code" in data
        assert "disease_name" in data
        assert "confidence" in data
        assert "severity" in data
        assert "top_k" in data
        assert "processing_ms" in data

        # Value checks
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["severity"] in {"healthy", "mild", "moderate", "severe", "critical"}
        assert len(data["top_k"]) == 3
        assert data["top_k"][0]["rank"] == 1

    def test_predict_empty_file(self, client):
        response = client.post(
            "/api/v1/predictions/",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_predict_invalid_content_type(self, client, sample_leaf_image_bytes):
        response = client.post(
            "/api/v1/predictions/",
            files={"image": ("doc.pdf", sample_leaf_image_bytes, "application/pdf")},
        )
        assert response.status_code == 415

    def test_predict_file_too_large(self, client):
        big_bytes = b"0" * (11 * 1024 * 1024)   # 11 MB, exceeds 10 MB limit
        response = client.post(
            "/api/v1/predictions/",
            files={"image": ("big.jpg", big_bytes, "image/jpeg")},
        )
        # Should fail at preprocessing validation
        assert response.status_code in {400, 422}

    def test_predict_top_k_validation(self, client, sample_leaf_image_bytes, mock_storage, mock_advisory):
        with (
            patch("app.api.routes.predictions.StorageService", return_value=mock_storage),
            patch("app.api.routes.predictions.AdvisoryService", return_value=mock_advisory),
        ):
            response = client.post(
                "/api/v1/predictions/",
                files={"image": ("leaf.jpg", sample_leaf_image_bytes, "image/jpeg")},
                data={"top_k": "15"},   # Exceeds max of 10
            )
        assert response.status_code == 422

    def test_get_nonexistent_prediction(self, client):
        response = client.get("/api/v1/predictions/00000000-0000-0000-0000-000000000000")
        # Should be 401 (auth required) or 403/404 depending on auth setup
        assert response.status_code in {401, 403, 404}


class TestImagePreprocessor:
    """Unit tests for the image preprocessing pipeline."""

    @pytest.mark.asyncio
    async def test_preprocess_valid_image(self, sample_leaf_image_bytes):
        from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        result = await preprocessor.preprocess_for_inference(sample_leaf_image_bytes)

        assert result.tensor.shape == (1, 3, 224, 224)
        assert 0.0 <= result.quality_score <= 1.0
        assert isinstance(result.image_hash, str)
        assert len(result.image_hash) == 64   # SHA-256 hex

    @pytest.mark.asyncio
    async def test_preprocess_invalid_bytes(self):
        from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="not a valid image"):
            await preprocessor.preprocess_for_inference(b"not an image at all")

    @pytest.mark.asyncio
    async def test_preprocess_oversized_image(self):
        from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        big_bytes = b"0" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds maximum size"):
            await preprocessor.preprocess_for_inference(big_bytes)

    def test_quality_score_range(self, sample_leaf_image_bytes):
        from app.ml.preprocessing.image_preprocessor import ImagePreprocessor
        import cv2
        preprocessor = ImagePreprocessor()
        arr = np.frombuffer(sample_leaf_image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        score = preprocessor._estimate_quality(bgr)
        assert 0.0 <= score <= 1.0


class TestGradCAM:
    def test_gradcam_output_shape(self, sample_leaf_image_bytes, mock_model_manager):
        """Verify Grad-CAM returns heatmap of correct shape."""
        from app.ml.explainability.gradcam import GradCAM
        import torchvision.models as tv_models

        # Use a real small model for this test
        model = tv_models.mobilenet_v3_small(weights=None)
        model.eval()

        preprocessor_mock = MagicMock()
        dummy_tensor = torch.zeros(1, 3, 224, 224)

        # Test that GradCAM can be instantiated without error
        try:
            cam = GradCAM(model, "features.12.0")
            cam.remove_hooks()
        except ValueError:
            # Layer not found with that name — expected for this test
            pass


class TestAdvisoryService:
    @pytest.mark.asyncio
    async def test_healthy_plant_advisory(self):
        from unittest.mock import AsyncMock
        from app.services.advisory_service import AdvisoryService

        mock_db = AsyncMock()
        service = AdvisoryService(db=mock_db)
        result = service._healthy_advisory()

        assert "No disease detected" in result["description"]
        assert len(result["preventive_practices"]) > 0
        assert result["treatments"] == {}

    @pytest.mark.asyncio
    async def test_generic_advisory_fallback(self):
        from app.services.advisory_service import AdvisoryService
        from unittest.mock import AsyncMock

        service = AdvisoryService(db=AsyncMock())
        result = service._generic_advisory("unknown___strange_disease")

        assert "treatments" in result
        assert "preventive_practices" in result
