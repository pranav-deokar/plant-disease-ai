# Plant Disease AI — Makefile
# Common development and deployment commands

.PHONY: help dev down db-migrate db-seed train test lint build push deploy logs clean

REGISTRY   ?= ghcr.io/yourorg
TAG        ?= $(shell git describe --tags --always 2>/dev/null || echo dev)
NAMESPACE  := plant-disease-ai
COMPOSE    := docker compose -f infrastructure/docker/docker-compose.yml

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────

dev:  ## Start full dev stack (all services + frontend)
	$(COMPOSE) --profile dev up -d
	@echo "\n✓ Stack running. API: http://localhost:8000/api/docs"

down:  ## Stop all services
	$(COMPOSE) down

restart:  ## Restart API and worker only
	$(COMPOSE) restart api worker

logs:  ## Tail API logs
	$(COMPOSE) logs -f api worker

logs-all:  ## Tail all service logs
	$(COMPOSE) logs -f

# ── Database ──────────────────────────────────────────────────────────────────

db-migrate:  ## Run Alembic migrations
	$(COMPOSE) exec api alembic upgrade head

db-rollback:  ## Roll back one migration
	$(COMPOSE) exec api alembic downgrade -1

db-seed:  ## Seed all data (PostgreSQL + MongoDB)
	$(COMPOSE) exec api python scripts/seed_postgres_diseases.py
	$(COMPOSE) exec api python scripts/seed_disease_kb.py
	@echo "✓ Seed complete"

db-shell:  ## Open PostgreSQL shell
	$(COMPOSE) exec db psql -U postgres plant_disease

# ── ML ────────────────────────────────────────────────────────────────────────

train:  ## Train EfficientNet-B4 model
	cd ml_pipeline && python training/train.py \
	  --model efficientnet_b4 \
	  --data-dir data/processed/plantvillage \
	  --epochs 30

train-fast:  ## Quick training run (5 epochs) for testing pipeline
	cd ml_pipeline && python training/train.py \
	  --model mobilenet_v3 --epochs 5 --batch-size 64

evaluate:  ## Evaluate model on test set
	cd ml_pipeline && python evaluation/evaluate.py \
	  --checkpoint ../backend/models/efficientnet_b4_best.pth \
	  --data-dir data/processed/plantvillage \
	  --output-dir reports/eval_$(TAG)

download-models:  ## Download pre-trained model weights
	$(COMPOSE) exec api python scripts/download_models.py --source url

# ── Testing ───────────────────────────────────────────────────────────────────

test:  ## Run all tests
	$(COMPOSE) exec api pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(COMPOSE) exec api pytest tests/ -v --cov=app --cov-report=html --cov-report=term
	@echo "Coverage report: backend/htmlcov/index.html"

test-fast:  ## Run tests (fail fast on first error)
	$(COMPOSE) exec api pytest tests/ -x -q

lint:  ## Run linting (ruff + mypy)
	$(COMPOSE) exec api ruff check app/ tests/
	$(COMPOSE) exec api mypy app/ --ignore-missing-imports

format:  ## Auto-format code
	$(COMPOSE) exec api ruff format app/ tests/

# ── Build & Deploy ────────────────────────────────────────────────────────────

build:  ## Build production Docker image
	docker build --target production \
	  -t $(REGISTRY)/plant-disease-api:$(TAG) \
	  -t $(REGISTRY)/plant-disease-api:latest \
	  backend/
	@echo "✓ Built $(REGISTRY)/plant-disease-api:$(TAG)"

push: build  ## Build and push to registry
	docker push $(REGISTRY)/plant-disease-api:$(TAG)
	docker push $(REGISTRY)/plant-disease-api:latest
	@echo "✓ Pushed $(REGISTRY)/plant-disease-api:$(TAG)"

deploy: push  ## Push and deploy to Kubernetes
	kubectl set image deployment/plant-disease-api \
	  api=$(REGISTRY)/plant-disease-api:$(TAG) \
	  -n $(NAMESPACE)
	kubectl set image deployment/plant-disease-worker \
	  worker=$(REGISTRY)/plant-disease-api:$(TAG) \
	  -n $(NAMESPACE)
	kubectl rollout status deployment/plant-disease-api -n $(NAMESPACE)
	@echo "✓ Deployed $(TAG) to $(NAMESPACE)"

rollback:  ## Roll back Kubernetes deployment
	kubectl rollout undo deployment/plant-disease-api -n $(NAMESPACE)
	kubectl rollout undo deployment/plant-disease-worker -n $(NAMESPACE)

health:  ## Check production health
	@curl -sf https://plantdisease.yourdomain.com/api/v1/health/ready | python -m json.tool

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:  ## Remove Docker volumes (DESTRUCTIVE — deletes all local data)
	@echo "WARNING: This will delete all local database data!"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	$(COMPOSE) down -v

shell:  ## Open a shell in the API container
	$(COMPOSE) exec api bash

preprocess:  ## Preprocess PlantVillage dataset
	cd ml_pipeline && python training/preprocess_dataset.py \
	  --input  data/raw/plantvillage_dataset/color \
	  --output data/processed/plantvillage \
	  --num-workers 8
