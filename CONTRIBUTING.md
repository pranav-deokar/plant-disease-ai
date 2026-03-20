# Contributing to Plant Disease AI

## Adding a New Disease Class

Follow these steps to add a new crop/disease to the detection system.

### 1. Collect images

Minimum **300 images** of the new disease, preferably:
- Taken in natural field conditions
- Multiple growth stages
- Various lighting conditions
- Multiple leaf angles (top, underside, stem)
- Cropped to show the leaf clearly (1:1 or 4:3 ratio)

Place images in:
```
ml_pipeline/data/raw/custom/{CropName}___{DiseaseName}/
```

Use the triple-underscore convention exactly as PlantVillage does:
```
ml_pipeline/data/raw/custom/Mango___Anthracnose/
ml_pipeline/data/raw/custom/Mango___healthy/
```

### 2. Update the disease class list

In `backend/app/ml/models/model_manager.py`, add to `DISEASE_CLASSES`:

```python
DISEASE_CLASSES: List[str] = [
    # ... existing classes ...
    "mango___anthracnose",    # new class
    "mango___healthy",        # corresponding healthy class
]
NUM_CLASSES = len(DISEASE_CLASSES)   # auto-updates
```

### 3. Update the folder mapping

In `ml_pipeline/training/preprocess_dataset.py`, add to `FOLDER_TO_CODE`:

```python
FOLDER_TO_CODE = {
    # ... existing mappings ...
    "Mango___Anthracnose": "mango___anthracnose",
    "Mango___healthy":     "mango___healthy",
}
```

### 4. Add disease metadata

In `backend/scripts/seed_postgres_diseases.py`, add to `DISEASE_META`:

```python
DISEASE_META = {
    # ... existing entries ...
    "mango___anthracnose": {
        "pathogen":  "fungal",
        "severity":  SeverityLevel.MODERATE,
        "impact":    "high",
        "contagious": True,
        "spread":    "fast",
    },
    "mango___healthy": {
        "pathogen":  None,
        "severity":  SeverityLevel.HEALTHY,
        "impact":    "low",
        "contagious": False,
        "spread":    "slow",
    },
}
```

### 5. Add knowledge base article

In `backend/scripts/seed_disease_kb.py`, add a full document to `DISEASE_DOCUMENTS`:

```python
{
    "disease_code": "mango___anthracnose",
    "display_name": "Mango — Anthracnose",
    "crop_name": "mango",
    "scientific_name": "Colletotrichum gloeosporioides",
    "pathogen_type": "fungal",
    "description": "Full disease description here...",
    "symptoms": ["Dark lesions on leaves", "Sunken spots on fruit", ...],
    "favorable_conditions": ["High humidity", "Warm temperatures (25-30°C)", ...],
    "economic_impact": "Can cause 20-80% post-harvest losses.",
    "treatments": {
        "chemical": [...],
        "organic": [...],
        "cultural": [...],
    },
    "preventive_practices": [...],
},
```

### 6. Retrain the model

```bash
# Reprocess combined dataset
make preprocess

# Fine-tune from existing checkpoint (faster than training from scratch)
python ml_pipeline/training/train.py \
  --model efficientnet_b4 \
  --data-dir ml_pipeline/data/processed/combined \
  --epochs 10 \
  --lr 5e-5 \
  --checkpoint backend/models/efficientnet_b4_plantvillage.pth
```

### 7. Evaluate and register

```bash
# Evaluate — check new class F1 score
make evaluate

# If test accuracy meets thresholds (>0.90 F1 on new class):
# Register via admin API or script
python backend/scripts/register_model.py \
  --name efficientnet_b4_plantvillage \
  --version v1.3.0 \
  --notes "Added mango anthracnose class"
```

### 8. Update seeds and redeploy

```bash
# Apply to running system
make db-seed    # updates PostgreSQL
# MongoDB seed is idempotent — safe to re-run anytime
docker compose exec api python scripts/seed_disease_kb.py

# Deploy new model
make deploy TAG=v1.3.0
```

---

## Code Style

- Python: follow PEP 8, formatted with `ruff format`
- TypeScript/JS: Prettier with single quotes, 2-space indent
- All new endpoints need at minimum a happy-path test
- Type hints required for all new Python functions

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New disease class added to all required locations
- [ ] Knowledge base article is complete and accurate
- [ ] Model re-evaluated with new class included
- [ ] DISEASE_CLASSES list is alphabetically sorted within crop groups
