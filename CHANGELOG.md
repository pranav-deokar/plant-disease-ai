# Changelog

All notable changes to the Plant Disease AI system are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Mobile app (React Native) with TFLite offline inference
- WhatsApp Bot integration via Twilio
- GPS-based disease outbreak heatmap
- Multilingual support (Hindi, Swahili, Spanish)

---

## [1.2.0] — 2025-06-01

### Added
- Grad-CAM++ explainability — disease region bounding boxes now included in response
- Disease severity score (0–1 continuous) alongside severity label
- `async_mode` parameter on POST /predictions — returns immediately, poll for result
- Rate limiting middleware with per-route limits and `X-RateLimit-*` response headers
- Prometheus custom metrics: confidence histogram, processing time histogram, model info gauge
- Admin: one-click model hot-swap via `POST /admin/models/{id}/activate`
- `GET /diseases/search` — full-text search across disease names and scientific names

### Changed
- EfficientNet-B4 updated to v1.2.0 weights (98.1% → 98.4% test accuracy)
- Leaf segmentation uses morphological cleanup for cleaner GrabCut masks
- Image quality score now includes greenness component (higher for actual leaf images)

### Fixed
- Prediction history pagination was off-by-one on last page
- Grad-CAM hooks not removed after inference (memory leak in long-running workers)
- CORS headers missing from 429 rate-limit responses

---

## [1.1.0] — 2025-03-15

### Added
- MobileNetV3-Large fallback model (95.7% accuracy, 120ms CPU inference)
- Per-user prediction statistics endpoint `GET /users/me/stats`
- API key management: create, list, revoke with scoped permissions
- Celery Beat scheduled tasks: retraining check (6h), image cleanup (daily), stats report (daily)
- MongoDB disease knowledge base with full symptoms, conditions, and treatment details
- `POST /predictions/{id}/feedback` — user correctness feedback for retraining data collection

### Changed
- Image preprocessing now uses CLAHE (adaptive contrast) instead of global histogram equalization
- Training augmentation pipeline expanded: added RandomPerspective, RandomErasing
- Advisory service now tries MongoDB first, falls back to PostgreSQL treatment_records

### Fixed
- GrabCut segmentation crashed on very small images (<64px) — now returns original image
- JWT refresh token race condition when multiple requests fired simultaneously on 401

---

## [1.0.0] — 2025-01-10

### Added
- Initial production release
- EfficientNet-B4 model trained on PlantVillage (38 classes, 54,306 images)
- FastAPI backend with async PostgreSQL (asyncpg) and Redis
- Image preprocessing pipeline: validation, GrabCut segmentation, CLAHE, normalization
- Grad-CAM++ disease region visualization
- Treatment recommendations (chemical, organic, cultural, biological)
- React frontend: upload, predict, view results, disease database browser
- Admin dashboard: system stats, model registry
- Docker Compose full stack (13 services)
- Kubernetes manifests with HPA (2–10 API pods, 1–8 worker pods)
- MLflow experiment tracking with per-run artifact storage
- Prometheus + Grafana monitoring with 9-panel dashboard
- Alembic database migrations
- CI/CD pipeline: test → build → scan → staging → production with manual approval gate

[Unreleased]: https://github.com/yourorg/plant-disease-ai/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/yourorg/plant-disease-ai/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/yourorg/plant-disease-ai/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/yourorg/plant-disease-ai/releases/tag/v1.0.0
