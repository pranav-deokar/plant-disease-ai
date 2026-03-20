"""
PostgreSQL Disease Seeder
──────────────────────────
Seeds the crop_diseases and treatment_records tables with
the 38 PlantVillage disease classes and their treatment data.

Run after applying migrations:
  python scripts/seed_postgres_diseases.py
"""
import os

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database_models import CropDisease, TreatmentRecord, SeverityLevel
from app.ml.models.model_manager import DISEASE_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@db:5432/plant_disease")
# Metadata for each class — pathogen type, severity, impact
DISEASE_META = {
    "apple___apple_scab":            {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "fast"},
    "apple___black_rot":             {"pathogen": "fungal",     "severity": SeverityLevel.SEVERE,   "impact": "high",   "contagious": True,  "spread": "moderate"},
    "apple___cedar_apple_rust":      {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": False, "spread": "slow"},
    "apple___healthy":               {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "blueberry___healthy":           {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "cherry___healthy":              {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "cherry___powdery_mildew":       {"pathogen": "fungal",     "severity": SeverityLevel.MILD,     "impact": "medium", "contagious": True,  "spread": "fast"},
    "corn___cercospora_leaf_spot":   {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "corn___common_rust":            {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "fast"},
    "corn___healthy":                {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "corn___northern_leaf_blight":   {"pathogen": "fungal",     "severity": SeverityLevel.SEVERE,   "impact": "high",   "contagious": True,  "spread": "fast"},
    "grape___black_rot":             {"pathogen": "fungal",     "severity": SeverityLevel.SEVERE,   "impact": "high",   "contagious": True,  "spread": "fast"},
    "grape___esca_black_measles":    {"pathogen": "fungal",     "severity": SeverityLevel.SEVERE,   "impact": "critical","contagious":False,  "spread": "slow"},
    "grape___healthy":               {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "grape___leaf_blight":           {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "orange___haunglongbing":        {"pathogen": "bacterial",  "severity": SeverityLevel.CRITICAL, "impact": "critical","contagious":True,   "spread": "fast"},
    "peach___bacterial_spot":        {"pathogen": "bacterial",  "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "peach___healthy":               {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "pepper___bacterial_spot":       {"pathogen": "bacterial",  "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "fast"},
    "pepper___healthy":              {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "potato___early_blight":         {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "moderate"},
    "potato___healthy":              {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "potato___late_blight":          {"pathogen": "oomycete",   "severity": SeverityLevel.CRITICAL, "impact": "critical","contagious":True,   "spread": "explosive"},
    "raspberry___healthy":           {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "soybean___healthy":             {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "squash___powdery_mildew":       {"pathogen": "fungal",     "severity": SeverityLevel.MILD,     "impact": "medium", "contagious": True,  "spread": "fast"},
    "strawberry___healthy":          {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "strawberry___leaf_scorch":      {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "tomato___bacterial_spot":       {"pathogen": "bacterial",  "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "fast"},
    "tomato___early_blight":         {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "moderate"},
    "tomato___healthy":              {"pathogen": None,         "severity": SeverityLevel.HEALTHY,  "impact": "low",    "contagious": False, "spread": "slow"},
    "tomato___late_blight":          {"pathogen": "oomycete",   "severity": SeverityLevel.CRITICAL, "impact": "critical","contagious":True,   "spread": "explosive"},
    "tomato___leaf_mold":            {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "tomato___mosaic_virus":         {"pathogen": "viral",      "severity": SeverityLevel.SEVERE,   "impact": "high",   "contagious": True,  "spread": "fast"},
    "tomato___septoria_leaf_spot":   {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "high",   "contagious": True,  "spread": "fast"},
    "tomato___spider_mites":         {"pathogen": "pest",       "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "fast"},
    "tomato___target_spot":          {"pathogen": "fungal",     "severity": SeverityLevel.MODERATE, "impact": "medium", "contagious": True,  "spread": "moderate"},
    "tomato___yellow_leaf_curl_virus":{"pathogen": "viral",     "severity": SeverityLevel.SEVERE,   "impact": "critical","contagious":True,   "spread": "fast"},
}

# Generic treatments by pathogen type
GENERIC_TREATMENTS = {
    "fungal": [
        {"type": "chemical",  "name": "Mancozeb",         "method": "Foliar spray",                     "dosage": "2.5g/L",       "freq": "Every 7–10 days"},
        {"type": "chemical",  "name": "Chlorothalonil",    "method": "Foliar spray",                     "dosage": "2g/L",         "freq": "Every 7–10 days"},
        {"type": "organic",   "name": "Copper fungicide",  "method": "Foliar spray (Bordeaux mixture)",  "dosage": "1% solution",  "freq": "Every 7 days"},
        {"type": "cultural",  "name": "Remove infected leaves", "method": "Prune and destroy infected material immediately", "dosage": None, "freq": "As needed"},
        {"type": "cultural",  "name": "Improve air circulation", "method": "Avoid dense planting; stake plants", "dosage": None, "freq": "Throughout season"},
    ],
    "bacterial": [
        {"type": "chemical",  "name": "Copper hydroxide",  "method": "Foliar spray",                     "dosage": "2g/L",         "freq": "Every 5–7 days"},
        {"type": "cultural",  "name": "Remove infected tissue", "method": "Prune with sterile tools; disinfect between cuts", "dosage": None, "freq": "As needed"},
        {"type": "cultural",  "name": "Avoid overhead irrigation", "method": "Use drip or furrow irrigation to keep foliage dry", "dosage": None, "freq": "Ongoing"},
        {"type": "organic",   "name": "Bacillus subtilis spray", "method": "Foliar preventive application", "dosage": "4L/ha",     "freq": "Every 5–7 days"},
    ],
    "viral": [
        {"type": "cultural",  "name": "Remove and destroy infected plants", "method": "Rogue out infected plants promptly — no chemical cure for viruses", "dosage": None, "freq": "As needed"},
        {"type": "cultural",  "name": "Control insect vectors", "method": "Control aphids, whiteflies, thrips with appropriate insecticides or traps", "dosage": None, "freq": "Throughout season"},
        {"type": "cultural",  "name": "Use certified virus-free seeds/transplants", "method": "Source from reputable certified suppliers", "dosage": None, "freq": "At planting"},
        {"type": "cultural",  "name": "Weed control", "method": "Remove weeds that serve as virus reservoirs", "dosage": None, "freq": "Regularly"},
    ],
    "oomycete": [
        {"type": "chemical",  "name": "Metalaxyl + Mancozeb", "method": "Foliar spray",                 "dosage": "2.5kg/ha",     "freq": "Every 7–10 days"},
        {"type": "chemical",  "name": "Cymoxanil + Mancozeb", "method": "Foliar spray (curative)",      "dosage": "2.0kg/ha",     "freq": "Every 7 days"},
        {"type": "organic",   "name": "Copper hydroxide",     "method": "Preventive foliar spray",       "dosage": "2–3kg/ha",     "freq": "Every 5–7 days"},
        {"type": "cultural",  "name": "Destroy infected material", "method": "Remove and burn/bury all infected haulm, tubers, and fruit", "dosage": None, "freq": "Immediately"},
    ],
    "pest": [
        {"type": "chemical",  "name": "Abamectin (miticide)", "method": "Foliar spray, thorough coverage of leaf undersides", "dosage": "1ml/L", "freq": "Every 7 days; max 3 applications"},
        {"type": "organic",   "name": "Neem oil spray",       "method": "Foliar spray including leaf undersides", "dosage": "5ml/L + surfactant", "freq": "Every 5–7 days"},
        {"type": "biological","name": "Phytoseiid predatory mites", "method": "Release 25–50 per m² for biological control", "dosage": None, "freq": "Once; monitor"},
        {"type": "cultural",  "name": "Overhead irrigation", "method": "Mite populations suppressed by wetting leaf surfaces", "dosage": None, "freq": "As needed"},
    ],
}


def format_display_name(code: str) -> str:
    parts = code.split("___")
    if len(parts) == 2:
        crop    = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title()
        return f"{crop} — {disease}"
    return code.replace("_", " ").title()


async def seed(db: AsyncSession):
    inserted_diseases = 0
    inserted_treatments = 0

    for idx, code in enumerate(DISEASE_CLASSES):
        # Check if already exists
        existing = await db.execute(
            select(CropDisease).where(CropDisease.disease_code == code)
        )
        if existing.scalar_one_or_none():
            logger.debug(f"Skipping existing: {code}")
            continue

        meta = DISEASE_META.get(code, {})
        crop_name = code.split("___")[0]
        is_healthy = code.endswith("_healthy")

        disease = CropDisease(
            id=uuid.uuid4(),
            disease_code=code,
            display_name=format_display_name(code),
            crop_name=crop_name,
            is_healthy=is_healthy,
            pathogen_type=meta.get("pathogen"),
            severity_default=meta.get("severity", SeverityLevel.MODERATE),
            is_contagious=meta.get("contagious", not is_healthy),
            spread_rate=meta.get("spread", "moderate"),
            economic_impact=meta.get("impact", "medium"),
            class_index=idx,
            training_samples=0,
            model_accuracy=None,
        )
        db.add(disease)
        await db.flush()   # get disease.id for FK
        inserted_diseases += 1

        # Add generic treatments based on pathogen type
        if not is_healthy and meta.get("pathogen"):
            treatments = GENERIC_TREATMENTS.get(meta["pathogen"], [])
            for t in treatments:
                treatment = TreatmentRecord(
                    id=uuid.uuid4(),
                    disease_id=disease.id,
                    treatment_type=t["type"],
                    treatment_name=t["name"],
                    application_method=t["method"],
                    dosage=t.get("dosage"),
                    frequency=t.get("freq"),
                    waiting_period_days=7 if t["type"] == "chemical" else None,
                    efficacy_score=None,
                    cost_level={"chemical": "medium", "organic": "low", "cultural": "low", "biological": "high"}.get(t["type"], "medium"),
                    availability={"chemical": "common", "organic": "common", "biological": "specialty"}.get(t["type"], "common"),
                )
                db.add(treatment)
                inserted_treatments += 1

    await db.commit()
    logger.info(f"Seeded {inserted_diseases} diseases and {inserted_treatments} treatment records")


async def main():
    engine = create_async_engine(DATABASE_URL, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        await seed(session)

    await engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
