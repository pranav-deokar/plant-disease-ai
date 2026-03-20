# Plant Disease AI — Deployment Runbook

## Overview

This document covers the full deployment lifecycle:
- Local development setup
- Staging deployment
- Production deployment on Kubernetes (EKS/GKE)
- Rollback procedures
- Incident response
- Scheduled maintenance

---

## Prerequisites

```bash
# Required tools
docker >= 24.0
kubectl >= 1.28
helm >= 3.14
aws-cli >= 2.15    # if deploying to AWS
python >= 3.11
node >= 20
```

---

## 1. Local Development

### First-time setup

```bash
git clone https://github.com/yourorg/plant-disease-ai.git
cd plant-disease-ai

# 1. Copy environment template
cp .env.example .env
# Edit .env — set SECRET_KEY at minimum

# 2. Start all infrastructure services
docker compose -f infrastructure/docker/docker-compose.yml --profile dev up -d

# 3. Wait for services to be healthy
docker compose ps   # all should show "healthy"

# 4. Run database migrations
docker compose exec api alembic upgrade head

# 5. Seed disease data
docker compose exec api python scripts/seed_postgres_diseases.py
docker compose exec api python scripts/seed_disease_kb.py

# 6. Download model weights (or train your own — see Training section)
docker compose exec api python scripts/download_models.py --source url

# 7. Verify health
curl http://localhost:8000/api/v1/health/ready
```

### Development URLs

| Service       | URL                          | Credentials          |
|--------------|------------------------------|----------------------|
| API Docs      | http://localhost:8000/api/docs | —                   |
| Frontend      | http://localhost:3000          | —                   |
| MLflow        | http://localhost:5000          | —                   |
| MinIO Console | http://localhost:9001          | minioadmin/minioadmin|
| Grafana       | http://localhost:3001          | admin/admin          |
| Flower        | http://localhost:5555          | admin/admin          |

### Running tests

```bash
# All tests
docker compose exec api pytest tests/ -v --cov=app --cov-report=term-missing

# Specific test file
docker compose exec api pytest tests/test_predictions.py -v

# With coverage report
docker compose exec api pytest tests/ --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 2. Model Training

### Prepare dataset

```bash
# Download PlantVillage from Kaggle
pip install kaggle
kaggle datasets download abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d ml_pipeline/data/raw/

# Preprocess (resize + deduplicate + organize)
cd ml_pipeline
python training/preprocess_dataset.py \
  --input  data/raw/plantvillage_dataset/color \
  --output data/processed/plantvillage \
  --num-workers 8

# Check preprocessing report
cat data/processed/plantvillage/preprocessing_report.json | python -m json.tool
```

### Train EfficientNet-B4

```bash
cd ml_pipeline

# Start MLflow server (if not running in Docker)
mlflow server --host 0.0.0.0 --port 5000 &

# Train (requires GPU for reasonable speed)
python training/train.py \
  --model efficientnet_b4 \
  --data-dir data/processed/plantvillage \
  --epochs 30 \
  --batch-size 32

# Expected training time:
#   RTX 4090:  ~4 hours
#   RTX 3080:  ~7 hours
#   A100:      ~2 hours
#   CPU only:  ~72 hours (not recommended)

# Evaluate on test set
python evaluation/evaluate.py \
  --checkpoint ../models/efficientnet_b4_best.pth \
  --data-dir data/processed/plantvillage \
  --output-dir reports/eval_v1
```

### Register model

```bash
# Copy weights to models directory
cp ml_pipeline/models/efficientnet_b4_best.pth backend/models/efficientnet_b4_plantvillage.pth

# Register in database
docker compose exec api python scripts/register_model.py \
  --name efficientnet_b4_plantvillage \
  --version v1.3.0 \
  --checkpoint models/efficientnet_b4_plantvillage.pth \
  --mlflow-run-id <run_id_from_mlflow>
```

---

## 3. Docker Build & Push

```bash
# Set your registry
REGISTRY=ghcr.io/yourorg
TAG=$(git describe --tags --always)

# Build production image
docker build \
  --target production \
  -t $REGISTRY/plant-disease-api:$TAG \
  -t $REGISTRY/plant-disease-api:latest \
  backend/

# Verify image
docker run --rm $REGISTRY/plant-disease-api:$TAG \
  python -c "import app.main; print('Import OK')"

# Push
docker push $REGISTRY/plant-disease-api:$TAG
docker push $REGISTRY/plant-disease-api:latest

# Build frontend
cd frontend
npm ci
npm run build
# dist/ is ready for nginx / S3 hosting
```

---

## 4. Kubernetes Deployment

### Initial cluster setup

```bash
# Create namespace
kubectl apply -f infrastructure/kubernetes/ingress.yaml

# Create secrets (use Vault or AWS Secrets Manager in production)
kubectl create secret generic plant-disease-secrets \
  --namespace plant-disease-ai \
  --from-literal=secret_key="$(openssl rand -hex 32)" \
  --from-literal=database_url="postgresql+asyncpg://user:pass@postgres-svc:5432/plant_disease" \
  --from-literal=mongodb_url="mongodb://mongo-svc:27017" \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy cert-manager for TLS
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true
```

### Deploy application

```bash
# Update image tag in manifests
sed -i "s|your-registry/plant-disease-api:v1.0.0|$REGISTRY/plant-disease-api:$TAG|g" \
  infrastructure/kubernetes/api-deployment.yaml \
  infrastructure/kubernetes/worker-deployment.yaml

# Apply all manifests
kubectl apply -f infrastructure/kubernetes/

# Run database migrations
kubectl run migrations --image=$REGISTRY/plant-disease-api:$TAG \
  --restart=Never --namespace=plant-disease-ai \
  --env-from=secretRef:plant-disease-secrets \
  --env-from=configMapRef:plant-disease-config \
  -- alembic upgrade head

# Wait for rollout
kubectl rollout status deployment/plant-disease-api -n plant-disease-ai
kubectl rollout status deployment/plant-disease-worker -n plant-disease-ai

# Verify
kubectl get pods -n plant-disease-ai
curl https://plantdisease.yourdomain.com/api/v1/health/ready
```

### Zero-downtime rolling update

```bash
# Update image tag
kubectl set image deployment/plant-disease-api \
  api=$REGISTRY/plant-disease-api:$NEW_TAG \
  -n plant-disease-ai

# Monitor rollout
kubectl rollout status deployment/plant-disease-api -n plant-disease-ai --timeout=5m

# Verify new version
kubectl get pods -n plant-disease-ai -l app=plant-disease-api -o wide
```

---

## 5. Rollback Procedures

### Code rollback (Kubernetes)

```bash
# Immediate rollback to previous deployment
kubectl rollout undo deployment/plant-disease-api -n plant-disease-ai
kubectl rollout undo deployment/plant-disease-worker -n plant-disease-ai

# Rollback to specific revision
kubectl rollout history deployment/plant-disease-api -n plant-disease-ai
kubectl rollout undo deployment/plant-disease-api --to-revision=3 -n plant-disease-ai
```

### Model rollback (hot-swap)

```bash
# Via admin API — promotes a previous model version
curl -X POST https://plantdisease.yourdomain.com/api/v1/admin/models/{model_id}/activate \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Or directly via CLI
kubectl exec -it deploy/plant-disease-api -n plant-disease-ai -- \
  python scripts/activate_model.py --model-id <uuid>
```

### Database rollback

```bash
# Roll back one migration
kubectl run db-rollback --image=$REGISTRY/plant-disease-api:$CURRENT_TAG \
  --restart=Never --namespace=plant-disease-ai \
  -- alembic downgrade -1

# Roll back to specific revision
kubectl run db-rollback --image=$REGISTRY/plant-disease-api:$CURRENT_TAG \
  --restart=Never --namespace=plant-disease-ai \
  -- alembic downgrade 001_initial
```

---

## 6. Monitoring & Alerting

### Key metrics to watch

| Metric | Warning | Critical |
|--------|---------|----------|
| API error rate | >1% | >5% |
| p95 prediction latency | >3s | >10s |
| Celery queue depth | >100 | >500 |
| Model avg confidence | <0.70 | <0.55 |
| DB connection pool | >80% | >95% |
| Redis memory | >70% | >90% |

### Check system health

```bash
# Health endpoints
curl https://plantdisease.yourdomain.com/api/v1/health
curl https://plantdisease.yourdomain.com/api/v1/health/ready

# Kubernetes pod status
kubectl get pods -n plant-disease-ai
kubectl describe pod <pod-name> -n plant-disease-ai

# API logs (last 100 lines)
kubectl logs -l app=plant-disease-api -n plant-disease-ai --tail=100 -f

# Worker logs
kubectl logs -l app=plant-disease-worker -n plant-disease-ai --tail=100 -f
```

### Celery queue monitoring

```bash
# Queue depths
kubectl exec deploy/plant-disease-worker -n plant-disease-ai -- \
  celery -A app.tasks.celery_app inspect active_queues

# Flush stuck predictions queue (emergency only)
kubectl exec deploy/plant-disease-worker -n plant-disease-ai -- \
  celery -A app.tasks.celery_app purge -Q predictions
```

---

## 7. Scheduled Maintenance

### Weekly tasks (automated via Celery Beat)

- Model retraining check (every 6 hours)
- Old image cleanup — S3 objects >90 days (daily at 2am)
- Admin stats report (daily at 6am)

### Monthly manual tasks

```bash
# 1. Review MLflow for new model candidates
open http://mlflow.internal:5000

# 2. Review per-class accuracy — look for degraded classes
python ml_pipeline/evaluation/evaluate.py \
  --checkpoint models/efficientnet_b4_plantvillage.pth \
  --data-dir ml_pipeline/data/processed/plantvillage \
  --output-dir reports/monthly_eval

# 3. Check feedback accuracy rate
curl https://plantdisease.yourdomain.com/api/v1/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq .correct_feedback_rate

# 4. Update disease knowledge base if new treatments published
python backend/scripts/seed_disease_kb.py   # idempotent upsert

# 5. Rotate API keys older than 90 days (notify users first)
# 6. Review and archive predictions older than 180 days
# 7. Check S3 storage costs — adjust cleanup policy if needed
```

---

## 8. Incident Response

### P1: All predictions failing

```bash
# 1. Check API health
curl https://plantdisease.yourdomain.com/api/v1/health/ready

# 2. Check model loaded
kubectl exec deploy/plant-disease-api -n plant-disease-ai -- \
  python -c "from app.ml.models.model_manager import ModelManager; print('OK')"

# 3. Check DB connectivity
kubectl exec deploy/plant-disease-api -n plant-disease-ai -- \
  python -c "import asyncio; from app.db.database import init_db; asyncio.run(init_db())"

# 4. Restart pods if stuck
kubectl rollout restart deployment/plant-disease-api -n plant-disease-ai
```

### P2: Model confidence degradation

```bash
# Check recent prediction stats
curl https://plantdisease.yourdomain.com/api/v1/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq '.avg_confidence, .correct_feedback_rate'

# If confidence < 0.65 consistently, trigger emergency retraining
curl -X POST https://plantdisease.yourdomain.com/api/v1/admin/retrain \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 9. Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | ✓ | 32-byte random hex — JWT signing key |
| `DATABASE_URL` | ✓ | PostgreSQL asyncpg connection string |
| `MONGODB_URL` | ✓ | MongoDB connection string |
| `REDIS_URL` | ✓ | Redis connection URL |
| `S3_BUCKET_NAME` | ✓ | S3/MinIO bucket for images |
| `AWS_ACCESS_KEY_ID` | Prod | AWS credentials (use IRSA in EKS) |
| `AWS_SECRET_ACCESS_KEY` | Prod | AWS credentials |
| `S3_ENDPOINT_URL` | Dev | MinIO endpoint URL |
| `SENTRY_DSN` | Prod | Sentry error tracking DSN |
| `MLFLOW_TRACKING_URI` | ✓ | MLflow server URL |
| `MODEL_CONFIDENCE_THRESHOLD` | — | Default: 0.65 |
| `CORS_ORIGINS` | ✓ | Comma-separated allowed origins |
