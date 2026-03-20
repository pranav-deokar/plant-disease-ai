"""
Advisory Service
─────────────────
Fetches disease descriptions and treatment recommendations from the
MongoDB knowledge base. Falls back to the PostgreSQL treatment_records
table if MongoDB is unavailable.

Knowledge base document structure (MongoDB):
{
  "_id": ObjectId,
  "disease_code": "tomato___early_blight",
  "display_name": "Tomato — Early Blight",
  "description": "...",
  "causal_organism": "Alternaria solani",
  "symptoms": ["circular brown spots", "target ring pattern", ...],
  "favorable_conditions": ["high humidity", "warm temperatures"],
  "economic_impact": "Can cause 50-78% yield loss if untreated",
  "treatments": {
    "chemical": [...],
    "organic": [...],
    "cultural": [...],
    "biological": [...],
  },
  "preventive_practices": ["crop rotation", "resistant varieties"],
  "images_reference": ["url1", "url2"],
  "last_updated": ISODate,
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.db.mongo import get_mongo_db

logger = logging.getLogger(__name__)


class AdvisoryService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache: Dict[str, dict] = {}   # simple in-process cache
    
    async def _get_ai_advisory(self, disease_code: str) -> dict:
        try:
            import requests
            import json
            import os

            prompt = f"""
            You are an agricultural expert.

            Provide treatment recommendations for:
            {disease_code}

            Return STRICT JSON format:
            {{
            "description": "...",
            "causal_organism": "...",
            "symptoms": ["..."],
            "favorable_conditions": ["..."],
            "economic_impact": "...",
            "treatments": {{
                "organic": [{{"treatment_name": "...", "application_method": "...", "dosage": "..."}}],
                "chemical": [{{"treatment_name": "...", "application_method": "...", "dosage": "..."}}]
            }},
            "preventive_practices": ["..."]
            }}
            """
            
            print("API KEY:", os.getenv("OPENROUTER_API_KEY"))

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",  # 🔥 good + cheap
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                },
            )

            if response.status_code != 200:
                print("OpenRouter ERROR RESPONSE:", response.text)
                raise Exception("OpenRouter request failed")

            result = response.json()

            # 🔍 Debug print (VERY IMPORTANT)
            print("OpenRouter RAW RESPONSE:", result)

            # Safe extraction
            choices = result.get("choices")
            if not choices:
                raise Exception(f"No choices in response: {result}")

            content = choices[0]["message"]["content"]

            # 🧠 Fix: extract JSON if wrapped in text
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)

            if not match:
                raise Exception("No JSON found in AI response")

            clean_json = match.group(0)

            return json.loads(clean_json)

        except Exception as e:
            logger.error(f"OpenRouter AI failed: {e}")
            print(f"OpenRouter AI failed: {e}")
            return self._generic_advisory(disease_code)
        
    
    async def get_treatments(self, disease_code: str,mode: str = "db") -> dict:
        """
        Return full advisory for a given disease code.
        Structure:
        {
          "description": str,
          "symptoms": list[str],
          "causal_organism": str,
          "favorable_conditions": list[str],
          "economic_impact": str,
          "treatments": {
            "chemical": [...],
            "organic": [...],
            "cultural": [...],
            "biological": [...],
          },
          "preventive_practices": list[str],
        }
        """
        if disease_code.endswith("healthy"):
            return self._healthy_advisory()
        # 🔥 AI MODE (LLM)
        if mode == "ai":
            try:
                return await self._get_ai_advisory(disease_code)
            except Exception as e:
                logger.warning(f"LLM advisory failed, falling back to DB: {e}")

        # Try cache first
        if disease_code in self._cache:
            return self._cache[disease_code]

        # Try MongoDB (rich data)
        try:
            advisory = await self._fetch_from_mongo(disease_code)
            if advisory:
                self._cache[disease_code] = advisory
                return advisory
        except Exception as e:
            logger.warning(f"MongoDB advisory lookup failed for {disease_code}: {e}")

        # Fallback to PostgreSQL treatment_records
        try:
            advisory = await self._fetch_from_postgres(disease_code)
            if advisory:
                self._cache[disease_code] = advisory
                return advisory
        except Exception as e:
            logger.warning(f"PostgreSQL advisory lookup failed for {disease_code}: {e}")

        # Final fallback: generic advice
        return self._generic_advisory(disease_code)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def _fetch_from_mongo(self, disease_code: str) -> Optional[dict]:
        mongo: AsyncIOMotorDatabase = await get_mongo_db()
        doc = await mongo.diseases.find_one({"disease_code": disease_code})
        if doc is None:
            return None

        # Strip MongoDB internal fields
        doc.pop("_id", None)
        return doc

    async def _fetch_from_postgres(self, disease_code: str) -> Optional[dict]:
        from app.models.database_models import CropDisease, TreatmentRecord
        from sqlalchemy.orm import selectinload

        result = await self.db.execute(
            select(CropDisease)
            .where(CropDisease.disease_code == disease_code)
            .options(selectinload(CropDisease.treatments))
        )
        disease = result.scalar_one_or_none()
        if disease is None:
            return None

        treatments_by_type: dict = {}
        for t in disease.treatments:
            entry = {
                "treatment_name": t.treatment_name,
                "active_ingredient": t.active_ingredient,
                "application_method": t.application_method,
                "dosage": t.dosage,
                "frequency": t.frequency,
                "waiting_period_days": t.waiting_period_days,
                "efficacy_score": t.efficacy_score,
                "cost_level": t.cost_level,
            }
            treatments_by_type.setdefault(t.treatment_type, []).append(entry)

        return {
            "description": f"{disease.display_name} caused by {disease.pathogen_type or 'unknown pathogen'}.",
            "causal_organism": disease.scientific_name,
            "symptoms": [],
            "favorable_conditions": [],
            "economic_impact": f"Economic impact: {disease.economic_impact or 'unknown'}",
            "treatments": treatments_by_type,
            "preventive_practices": [],
        }


    def _healthy_advisory(self) -> dict:
        return {
            "description": "No disease detected. The plant appears healthy.",
            "causal_organism": None,
            "symptoms": [],
            "favorable_conditions": [],
            "economic_impact": "None",
            "treatments": {},
            "preventive_practices": [
                "Continue regular monitoring of leaves and stems.",
                "Maintain proper irrigation — avoid waterlogging.",
                "Use balanced fertilization based on soil test results.",
                "Practice crop rotation to reduce pathogen buildup.",
                "Remove and destroy any diseased plant material promptly.",
            ],
        }

    def _generic_advisory(self, disease_code: str) -> dict:
        """Fallback when no specific advice is found in any data store."""
        return {
            "description": f"Disease identified: {disease_code.replace('___', ' — ').replace('_', ' ').title()}",
            "causal_organism": "Unknown",
            "symptoms": ["Visible damage on leaf surface"],
            "favorable_conditions": ["Check local agricultural extension resources"],
            "economic_impact": "Variable",
            "treatments": {
                "general": [
                    {
                        "treatment_name": "Contact local agricultural extension",
                        "application_method": "Consult a certified agronomist for region-specific treatment advice.",
                        "dosage": "N/A",
                    }
                ]
            },
            "preventive_practices": [
                "Remove infected plant material immediately.",
                "Improve air circulation around plants.",
                "Avoid overhead irrigation.",
                "Use certified disease-free seeds.",
            ],
        }

    async def get_disease_info(self, disease_code: str) -> Optional[dict]:
        """Return the full disease knowledge article."""
        try:
            mongo: AsyncIOMotorDatabase = await get_mongo_db()
            doc = await mongo.diseases.find_one({"disease_code": disease_code})
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception as e:
            logger.warning(f"Failed to fetch disease info from MongoDB: {e}")
        return None

    async def list_diseases(
        self, crop_name: Optional[str] = None, skip: int = 0, limit: int = 50
    ) -> List[dict]:
        """List all diseases, optionally filtered by crop."""
        try:
            mongo: AsyncIOMotorDatabase = await get_mongo_db()
            query = {}
            if crop_name:
                query["crop_name"] = crop_name.lower()
            cursor = mongo.diseases.find(
                query, {"disease_code": 1, "display_name": 1, "crop_name": 1, "pathogen_type": 1}
            ).skip(skip).limit(limit)
            return [
                {k: v for k, v in doc.items() if k != "_id"}
                async for doc in cursor
            ]
        except Exception as e:
            logger.error(f"Failed to list diseases: {e}")
            return []
