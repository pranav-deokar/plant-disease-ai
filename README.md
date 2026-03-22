# 🌿 AI-Based Plant Disease Detection & Advisory System

A production-grade, full-stack AI system for detecting plant diseases from leaf images
and providing structured treatment recommendations to farmers.

---

## 🚀 Live Demo

🌐 Frontend: https://plant-disease-ai1-pranav-deokars-projects.vercel.app/ 
⚙️ Backend API: https://plant-disease-ai-5qyh.onrender.com/api/v1/health

⚠️ Note:
Backend is hosted on Render free tier.  
IMPORTANT: THE BACKEND TAKES TIME TO WAKEUP 
If it is inactive:

1. Open backend link first  
2. Wait 20–30 seconds
   (   Until you see
      {"status":"ok","service":"Plant Disease AI","version":"1.0.0","uptime_seconds":18}
   )
4. Then use frontend  




## Table of Contents

1. [System Architecture](#system-architecture)
2. [Folder Structure](#folder-structure)
3. [Technology Stack](#technology-stack)
4. [Quick Start (Docker)](#quick-start)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Training](#model-training)
7. [API Reference](#api-reference)
8. [Deployment (Production)](#deployment)
9. [Monitoring](#monitoring)
10. [Future Improvements](#future-improvements)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                        │
│        React Web App | Mobile View | Admin Dashboard    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS / REST
┌────────────────────────▼────────────────────────────────┐
│             API GATEWAY  (FastAPI)                       │
│        Auth · Rate Limiting · Request Routing            │
└──────────┬──────────────┬──────────────────┬────────────┘
           │              │                  │
   ┌───────▼──┐   ┌───────▼──┐   ┌──────────▼─┐
   │  Image   │   │Prediction│   │  Advisory  │
   │ Service  │   │  Engine  │   │  Engine    │
   └───────┬──┘   └───┬──────┘   └────────────┘
           │          │ Grad-CAM · Severity
   ┌───────▼──┐   ┌───▼──────────────────┐
   │  Object  │   │    ML Model Server   │
   │ Storage  │   │  EfficientNet-B4     │
   │  (S3)    │   │  MobileNetV3         │
   └──────────┘   └──────────────────────┘
           │          │
   ┌───────▼──────────▼──────────────────┐
   │          DATA LAYER                  │
   │  PostgreSQL · MongoDB · Redis         │
   │  Elasticsearch                        │
   └──────────────────────────────────────┘
           │
   ┌───────▼──────────────────────────────┐
   │          MLOps PIPELINE               │
   │  MLflow · DVC · Celery · Retraining   │
   └──────────────────────────────────────┘
```

---

## Folder Structure

```
plant_disease_ai/
│
├── backend/                        # FastAPI backend
│   ├── app/
│   │   ├── main.py                 # Application entry point
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic settings
│   │   │   ├── logging.py          # Structured logging setup
│   │   │   └── security.py         # JWT, hashing utilities
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── predictions.py  # ⭐ Core prediction API
│   │   │   │   ├── diseases.py     # Disease knowledge base API
│   │   │   │   ├── auth.py         # Login, register, refresh
│   │   │   │   ├── users.py        # User profile management
│   │   │   │   ├── admin.py        # Admin dashboard routes
│   │   │   │   └── health.py       # /health liveness probe
│   │   │   └── dependencies.py     # FastAPI dependency injection
│   │   ├── models/
│   │   │   └── database_models.py  # SQLAlchemy ORM models
│   │   ├── schemas/
│   │   │   ├── prediction_schemas.py
│   │   │   └── user_schemas.py
│   │   ├── services/
│   │   │   ├── prediction_service.py  # ⭐ Inference orchestration
│   │   │   ├── advisory_service.py    # Treatment recommendations
│   │   │   └── storage_service.py     # S3 / MinIO operations
│   │   ├── ml/
│   │   │   ├── models/
│   │   │   │   └── model_manager.py   # ⭐ Model loading + serving
│   │   │   ├── preprocessing/
│   │   │   │   └── image_preprocessor.py  # ⭐ CV pipeline
│   │   │   └── explainability/
│   │   │       └── gradcam.py         # ⭐ Grad-CAM++
│   │   ├── db/
│   │   │   ├── database.py            # Async SQLAlchemy engine
│   │   │   └── mongo.py               # Motor MongoDB client
│   │   ├── tasks/
│   │   │   ├── celery_app.py          # Celery configuration
│   │   │   └── prediction_tasks.py    # Async prediction tasks
│   │   └── utils/
│   │       └── image_utils.py
│   ├── tests/
│   │   ├── test_predictions.py
│   │   ├── test_preprocessing.py
│   │   └── conftest.py
│   ├── migrations/                    # Alembic migrations
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                          # React frontend
│   └── src/
│       ├── pages/
│       │   ├── PredictPage.jsx        # ⭐ Main upload/predict UI
│       │   ├── HistoryPage.jsx
│       │   ├── DiseasesPage.jsx
│       │   └── AdminPage.jsx
│       ├── components/
│       │   ├── disease/
│       │   │   ├── PredictionCard.jsx
│       │   │   ├── TreatmentPanel.jsx
│       │   │   └── GradCAMViewer.jsx
│       │   ├── common/
│       │   │   ├── ImageDropzone.jsx
│       │   │   └── ConfidenceBar.jsx
│       │   └── admin/
│       │       └── StatsDashboard.jsx
│       ├── hooks/
│       │   ├── usePrediction.js
│       │   └── useAuth.js
│       └── services/
│           └── api.js
│
├── ml_pipeline/                       # Offline ML work
│   ├── data/
│   │   ├── raw/                       # PlantVillage original zips
│   │   ├── processed/                 # Resized, organized images
│   │   └── augmented/                 # Augmented training set
│   ├── notebooks/
│   │   ├── 01_eda.ipynb               # Exploratory data analysis
│   │   ├── 02_baseline.ipynb          # Baseline model experiments
│   │   └── 03_explainability.ipynb    # Grad-CAM analysis
│   ├── training/
│   │   └── train.py                   # ⭐ Full training script
│   ├── evaluation/
│   │   └── evaluate.py                # Per-class metrics, confusion matrix
│   └── configs/
│       ├── efficientnet_b4.yaml
│       └── mobilenet_v3.yaml
│
└── infrastructure/
    ├── docker/
    │   └── docker-compose.yml         # ⭐ Full stack compose
    ├── kubernetes/
    │   ├── api-deployment.yaml
    │   ├── worker-deployment.yaml
    │   └── ingress.yaml
    ├── nginx/
    │   └── nginx.conf
    └── monitoring/
        ├── prometheus.yml
        └── grafana/
            └── dashboards/
                └── plant_disease.json
```

---

## Technology Stack

| Layer              | Technology                         | Purpose                               |
|--------------------|-------------------------------------|---------------------------------------|
| **ML Model**       | PyTorch + EfficientNet-B4           | Disease classification (38 classes)   |
| **Fallback Model** | MobileNetV3-Large                   | Edge / low-latency inference          |
| **Image Processing**| OpenCV + Pillow                    | Preprocessing, segmentation           |
| **Backend API**    | FastAPI + Uvicorn                   | High-performance async REST API       |
| **Primary DB**     | PostgreSQL (asyncpg)                | Users, predictions, history           |
| **Knowledge Base** | MongoDB (Motor)                     | Disease articles, treatments          |
| **Cache**          | Redis                               | Prediction caching, session store     |
| **Object Storage** |      MinIO                          | Images, Grad-CAM overlays             |
| **Experiment Tracking** | MLflow                         | Training runs, model registry         |
| **Frontend**       | React 18 + TailwindCSS              | Farmer UI                             |
| **Infrastructure** | Docker                              | Containerized deployment              |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.24
- NVIDIA GPU + CUDA 12.1 (optional, for GPU acceleration)

### 1. Clone and configure

```bash
git clone https://github.com/yourorg/plant-disease-ai.git
cd plant-disease-ai

# Copy and edit environment variables
cp .env.example .env
# Required: set SECRET_KEY, POSTGRES_PASSWORD
```

### 2. Start all services

```bash
# Development (with hot-reload + frontend dev server)
docker compose -f infrastructure/docker/docker-compose.yml --profile dev up -d

# Production (no dev server, optimized builds)
docker compose -f infrastructure/docker/docker-compose.yml up -d
```

### 3. Initialize the database

```bash
# Run migrations
docker compose exec api alembic upgrade head

# Seed the disease knowledge base
docker compose exec api python scripts/seed_disease_kb.py
```

### 4. Download and place model weights

```bash
# Option A: Download pre-trained weights
docker compose exec api python scripts/download_models.py

# Option B: Train your own (see Model Training section)
```

### 5. Access the system

| Service          | URL                      |
|-----------------|--------------------------|
| Web App          | http://localhost:3000    |
| API Docs         | http://localhost:8000/api/docs |
| MLflow           | http://localhost:5000    |
| MinIO Console    | http://localhost:9001    |
| Grafana          | http://localhost:3001    |
| Flower (Celery)  | http://localhost:5555    |

---

## Dataset Preparation

### PlantVillage Dataset

```bash
# Download from Kaggle (requires Kaggle API key)
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d ml_pipeline/data/raw/

# Preprocess: resize + organize
python ml_pipeline/training/preprocess_dataset.py \
  --input ml_pipeline/data/raw/plantvillage_dataset/color \
  --output ml_pipeline/data/processed/plantvillage \
  --num-workers 8
```

**Dataset stats (PlantVillage):**
- 54,306 images across 38 classes
- 14 crop types
- 26 disease categories + 12 healthy classes
- Class imbalance range: 152–5,507 images per class

### Custom Dataset Addition

```bash
# Add new crop/disease data:
# 1. Create directory: ml_pipeline/data/raw/custom/{CropName}___{DiseaseName}/
# 2. Add minimum 200 high-quality images
# 3. Re-run preprocessing
# 4. Update DISEASE_CLASSES in model_manager.py
# 5. Retrain with fine-tuning from existing checkpoint
```

---

## Model Training

```bash
# Install training dependencies (local, not Docker)
pip install -r backend/requirements.txt

# Train EfficientNet-B4 (recommended, ~4h on RTX 4090)
python ml_pipeline/training/train.py \
  --model efficientnet_b4 \
  --data-dir ml_pipeline/data/processed/plantvillage \
  --epochs 30 \
  --batch-size 32

# Train MobileNetV3 (faster, lighter, ~1h on RTX 4090)
python ml_pipeline/training/train.py \
  --model mobilenet_v3 \
  --batch-size 64 \
  --epochs 20

# Monitor training live in MLflow
open http://localhost:5000
```

**Expected performance (PlantVillage test set):**

| Model           | Accuracy | F1 Macro | Inference (CPU) | Inference (GPU) |
|----------------|----------|----------|-----------------|-----------------|
| EfficientNet-B4 | 98.1%    | 97.9%    | ~850ms          | ~45ms           |
| MobileNetV3-L   | 95.7%    | 95.3%    | ~120ms          | ~15ms           |

---

## API Reference

### POST `/api/v1/predictions/`

Submit a leaf image for disease prediction.

**Request (multipart/form-data):**
```
image: <file>           # Required. JPEG/PNG/WebP, max 10MB
crop_hint: "tomato"     # Optional. Helps guide prediction
top_k: 5                # Optional. Number of alternatives (1-10)
async_mode: false       # Optional. Return immediately, poll for result
```

**Response:**
```json
{
  "prediction_id": "uuid-...",
  "disease_code": "tomato___early_blight",
  "disease_name": "Tomato — Early Blight",
  "confidence": 0.9423,
  "severity": "moderate",
  "severity_score": 0.38,
  "top_k": [
    { "rank": 1, "disease_code": "tomato___early_blight", "confidence": 0.9423 },
    { "rank": 2, "disease_code": "tomato___target_spot",  "confidence": 0.0421 }
  ],
  "gradcam_url": "https://s3.amazonaws.com/bucket/gradcam/uuid/overlay.jpg",
  "attention_boxes": [
    { "x": 120, "y": 80, "width": 200, "height": 150, "confidence": 0.87 }
  ],
  "image_quality_score": 0.82,
  "is_leaf_detected": true,
  "warnings": [],
  "processing_ms": 342,
  "treatments": [
    {
      "treatment_type": "chemical",
      "treatment_name": "Chlorothalonil",
      "application_method": "Foliar spray",
      "dosage": "2g per litre of water",
      "waiting_period_days": 7
    },
    {
      "treatment_type": "organic",
      "treatment_name": "Copper-based fungicide",
      "application_method": "Spray every 7-10 days"
    }
  ],
  "model_name": "efficientnet_b4_plantvillage",
  "model_version": "v1.2.0"
}
```

---

## Deployment (Production)

### Environment variables for production

```bash
# Required secrets (use K8s Secrets or Vault)
SECRET_KEY=<32-byte-random-string>
POSTGRES_PASSWORD=<strong-password>
MINIO_ROOT_PASSWORD=<strong-password>

# AWS (if using real S3 instead of MinIO)
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
S3_BUCKET_NAME=your-production-bucket

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
```

---

## Monitoring

Grafana dashboards available at http://localhost:3001 after startup:

| Dashboard              | Metrics                                       |
|-----------------------|-----------------------------------------------|
| Prediction Overview    | Requests/min, avg latency, error rate         |
| Model Performance      | Confidence distribution, class distribution   |
| System Health          | CPU, GPU memory, DB connections               |
| Business Metrics       | Daily predictions, top diseases detected      |

---

## Future Improvements

### Short-term (v1.x)
- [ ] Mobile app (React Native) with offline inference (TFLite / Core ML)
- [ ] Multilingual support (Hindi, Swahili, Spanish for target farmer demographics)
- [ ] GPS-based disease outbreak mapping
- [ ] Crop-specific fine-tuned models (tomato-only, potato-only etc.)
- [ ] WhatsApp Bot integration for low-tech farmers

### Medium-term (v2.x)
- [ ] Time-series analysis for disease progression tracking
- [ ] Drone imagery support (multispectral)
- [ ] Federated learning for privacy-preserving model updates from field devices
- [ ] Integration with weather APIs for disease risk forecasting
- [ ] Marketplace for certified agronomist consultations

### Long-term (v3.x)
- [ ] Foundation model fine-tuning (SAM, DINOv2) for better few-shot generalization
- [ ] Multimodal input (leaf image + soil data + weather + crop age)
- [ ] Autonomous treatment drone dispatch integration
- [ ] Carbon footprint optimization in treatment recommendations

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, coding standards,
and how to add new disease classes to the knowledge base.

## License

MIT License — see [LICENSE](LICENSE)
