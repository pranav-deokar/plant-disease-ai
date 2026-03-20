"""
Disease Database Routes
────────────────────────
GET  /api/v1/diseases/              — list all diseases (filterable)
GET  /api/v1/diseases/{code}        — full disease detail
GET  /api/v1/diseases/crops         — list available crop names
GET  /api/v1/diseases/search        — full-text search
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_optional_user
from app.models.database_models import CropDisease, User
from app.services.advisory_service import AdvisoryService

router = APIRouter()


@router.get("/", summary="List all plant diseases")
async def list_diseases(
    crop_name: Optional[str] = Query(None, description="Filter by crop (e.g. 'tomato')"),
    pathogen_type: Optional[str] = Query(None, description="fungal | bacterial | viral | pest"),
    include_healthy: bool = Query(False, description="Include 'Healthy' pseudo-classes"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    query = select(CropDisease)
    if crop_name:
        query = query.where(CropDisease.crop_name == crop_name.lower())
    if pathogen_type:
        query = query.where(CropDisease.pathogen_type == pathogen_type)
    if not include_healthy:
        query = query.where(CropDisease.is_healthy == False)
    query = query.order_by(CropDisease.crop_name, CropDisease.display_name)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    diseases = result.scalars().all()
    return [
        {
            "disease_code": d.disease_code,
            "display_name": d.display_name,
            "crop_name": d.crop_name,
            "pathogen_type": d.pathogen_type,
            "severity_default": d.severity_default.value if d.severity_default else None,
            "economic_impact": d.economic_impact,
            "is_contagious": d.is_contagious,
            "model_accuracy": d.model_accuracy,
        }
        for d in diseases
    ]


@router.get("/crops", summary="List all supported crop types")
async def list_crops(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func, distinct
    result = await db.execute(
        select(distinct(CropDisease.crop_name)).order_by(CropDisease.crop_name)
    )
    return {"crops": [row[0] for row in result.all()]}


@router.get("/search", summary="Search diseases by keyword")
async def search_diseases(
    q: str = Query(..., min_length=2, description="Search term"),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import or_
    result = await db.execute(
        select(CropDisease).where(
            or_(
                CropDisease.display_name.ilike(f"%{q}%"),
                CropDisease.disease_code.ilike(f"%{q}%"),
                CropDisease.scientific_name.ilike(f"%{q}%"),
            )
        ).limit(20)
    )
    diseases = result.scalars().all()
    return [{"disease_code": d.disease_code, "display_name": d.display_name, "crop_name": d.crop_name} for d in diseases]


@router.get("/{disease_code}", summary="Full disease details with treatments")
async def get_disease(
    disease_code: str,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(CropDisease)
        .where(CropDisease.disease_code == disease_code)
        .options(selectinload(CropDisease.treatments))
    )
    disease = result.scalar_one_or_none()
    if disease is None:
        raise HTTPException(status_code=404, detail=f"Disease '{disease_code}' not found.")

    # Enrich with MongoDB knowledge article
    advisory = AdvisoryService(db=db)
    knowledge = await advisory.get_disease_info(disease_code)

    return {
        "disease_code": disease.disease_code,
        "display_name": disease.display_name,
        "scientific_name": disease.scientific_name,
        "crop_name": disease.crop_name,
        "pathogen_type": disease.pathogen_type,
        "severity_default": disease.severity_default.value if disease.severity_default else None,
        "is_contagious": disease.is_contagious,
        "spread_rate": disease.spread_rate,
        "economic_impact": disease.economic_impact,
        "class_index": disease.class_index,
        "training_samples": disease.training_samples,
        "model_accuracy": disease.model_accuracy,
        "treatments": [
            {
                "treatment_type": t.treatment_type,
                "treatment_name": t.treatment_name,
                "active_ingredient": t.active_ingredient,
                "application_method": t.application_method,
                "dosage": t.dosage,
                "frequency": t.frequency,
                "waiting_period_days": t.waiting_period_days,
                "efficacy_score": t.efficacy_score,
                "cost_level": t.cost_level,
            }
            for t in sorted(disease.treatments, key=lambda x: x.treatment_type)
        ],
        "knowledge": knowledge or {},
    }
