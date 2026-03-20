"""
Disease Knowledge Base Seeder
──────────────────────────────
Seeds MongoDB with detailed disease information for all 38 PlantVillage classes.
Run once after initial deployment:
  python scripts/seed_disease_kb.py

Each disease document includes:
  - Full description and causal organism
  - Symptoms and favorable conditions
  - Chemical, organic, cultural, and biological treatments
  - Preventive farming practices
"""
import os
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URL = os.environ.get("MONGODB_URL", "mongodb://mongo:27017")
DB_NAME = "plant_disease_kb"

# ── Sample disease documents (showing 3 of 38 for brevity) ────────────────────
# In production, all 38 classes should be fully populated.
DISEASE_DOCUMENTS = [
    {
        "disease_code": "tomato___early_blight",
        "display_name": "Tomato — Early Blight",
        "crop_name": "tomato",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "fungal",
        "description": (
            "Early blight is one of the most common fungal diseases of tomato, "
            "caused by Alternaria solani. It affects leaves, stems, and fruit, "
            "and can cause significant yield losses if untreated. The disease "
            "typically appears on older, lower leaves first and progresses upward."
        ),
        "symptoms": [
            "Small, dark brown to black spots with a yellow halo on older leaves",
            "Characteristic concentric ring pattern ('target board' or 'bull's-eye' lesion)",
            "Spots enlarge and coalesce causing leaf yellowing and defoliation",
            "Dark, sunken lesions at the stem end of fruit (collar rot)",
            "Premature fruit drop in severe infections",
        ],
        "favorable_conditions": [
            "Warm temperatures (24–29°C / 75–84°F)",
            "High relative humidity (>80%) or leaf wetness lasting more than 2 hours",
            "Overhead irrigation promoting leaf wetness",
            "Dense canopy with poor air circulation",
            "Plants stressed by drought, nutrient deficiency, or other diseases",
        ],
        "economic_impact": (
            "Can cause 50–78% yield loss in susceptible varieties if untreated. "
            "Direct losses from fruit infection and indirect losses from defoliation "
            "reducing photosynthesis and sunscald exposure."
        ),
        "treatments": {
            "chemical": [
                {
                    "treatment_name": "Chlorothalonil (e.g., Bravo)",
                    "active_ingredient": "Chlorothalonil",
                    "application_method": "Foliar spray, thorough coverage of upper and lower leaf surfaces",
                    "dosage": "1.5–2.0 kg/ha or 2g per litre of water",
                    "frequency": "Every 7–10 days, begin at first sign of disease",
                    "waiting_period_days": 7,
                    "efficacy_score": 0.82,
                    "cost_level": "low",
                    "availability": "common",
                    "notes": "Broad-spectrum protectant. Do not exceed 12 applications per season.",
                },
                {
                    "treatment_name": "Mancozeb (e.g., Dithane M-45)",
                    "active_ingredient": "Mancozeb",
                    "application_method": "Foliar spray",
                    "dosage": "2.0 kg/ha or 2.5g per litre",
                    "frequency": "Every 7 days",
                    "waiting_period_days": 5,
                    "efficacy_score": 0.79,
                    "cost_level": "low",
                    "availability": "common",
                },
                {
                    "treatment_name": "Azoxystrobin + Difenoconazole (e.g., Amistar Top)",
                    "active_ingredient": "Azoxystrobin 20% + Difenoconazole 12.5%",
                    "application_method": "Foliar spray",
                    "dosage": "0.5–1.0 ml per litre of water",
                    "frequency": "Every 10–14 days; max 4 applications per season",
                    "waiting_period_days": 3,
                    "efficacy_score": 0.91,
                    "cost_level": "medium",
                    "availability": "common",
                    "notes": "Systemic + protectant combination. Rotate with other fungicide groups to prevent resistance.",
                },
            ],
            "organic": [
                {
                    "treatment_name": "Copper-based fungicide (e.g., Bordeaux mixture)",
                    "active_ingredient": "Copper sulfate + hydrated lime",
                    "application_method": "Foliar spray, 1% Bordeaux mixture",
                    "dosage": "10g copper sulfate + 10g lime per litre of water",
                    "frequency": "Every 7–10 days",
                    "waiting_period_days": 1,
                    "efficacy_score": 0.68,
                    "cost_level": "low",
                    "availability": "common",
                    "notes": "Avoid in hot weather to prevent phytotoxicity.",
                },
                {
                    "treatment_name": "Bacillus subtilis (e.g., Serenade)",
                    "active_ingredient": "Bacillus subtilis strain QST 713",
                    "application_method": "Foliar spray, preventive application",
                    "dosage": "4 litres per hectare",
                    "frequency": "Every 5–7 days",
                    "waiting_period_days": 0,
                    "efficacy_score": 0.61,
                    "cost_level": "medium",
                    "availability": "specialty",
                    "notes": "Best used preventively or at very early infection stages.",
                },
                {
                    "treatment_name": "Neem oil (1–2%)",
                    "active_ingredient": "Azadirachtin",
                    "application_method": "Foliar spray with surfactant",
                    "dosage": "10–20ml per litre of water + 2ml surfactant",
                    "frequency": "Every 7 days",
                    "waiting_period_days": 0,
                    "efficacy_score": 0.55,
                    "cost_level": "low",
                    "availability": "common",
                },
            ],
            "cultural": [
                {
                    "treatment_name": "Crop rotation",
                    "application_method": "Do not plant tomato or other Solanaceae in the same field for 2–3 years",
                    "efficacy_score": 0.75,
                    "cost_level": "low",
                },
                {
                    "treatment_name": "Remove infected plant material",
                    "application_method": "Prune and destroy (burn or deep bury) infected leaves and debris immediately. Do not compost.",
                    "efficacy_score": 0.70,
                    "cost_level": "low",
                },
                {
                    "treatment_name": "Drip irrigation",
                    "application_method": "Switch from overhead to drip or furrow irrigation to keep foliage dry",
                    "efficacy_score": 0.65,
                    "cost_level": "medium",
                },
                {
                    "treatment_name": "Stake and prune for air circulation",
                    "application_method": "Remove lower leaves touching soil. Stake plants to improve canopy airflow.",
                    "efficacy_score": 0.60,
                    "cost_level": "low",
                },
            ],
            "biological": [
                {
                    "treatment_name": "Trichoderma harzianum soil drench",
                    "application_method": "Apply as soil drench at transplanting and 4 weeks later",
                    "dosage": "5g per litre of water",
                    "frequency": "At transplanting + monthly",
                    "efficacy_score": 0.58,
                    "cost_level": "medium",
                    "availability": "specialty",
                },
            ],
        },
        "preventive_practices": [
            "Use certified disease-free or treated seed.",
            "Plant resistant/tolerant varieties (e.g., Mountain Magic, Plum Regal).",
            "Maintain soil fertility — adequate phosphorus and potassium strengthen cell walls.",
            "Mulch around plants to prevent soil splash inoculation.",
            "Water at the base of plants, early in the day so foliage dries before evening.",
            "Apply preventive fungicide sprays from transplanting through fruiting in high-risk conditions.",
            "Monitor daily and scout lower leaves for early lesions.",
        ],
        "resistant_varieties": ["Mountain Magic", "Plum Regal", "Defiant PhR", "Iron Lady"],
    },

    {
        "disease_code": "potato___late_blight",
        "display_name": "Potato — Late Blight",
        "crop_name": "potato",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "oomycete",
        "description": (
            "Late blight is the most destructive disease of potato and tomato worldwide, "
            "caused by the oomycete Phytophthora infestans. It was responsible for the "
            "Irish Potato Famine (1845–1852). Under favorable conditions, entire fields "
            "can be destroyed within days. The pathogen spreads rapidly via airborne sporangia."
        ),
        "symptoms": [
            "Water-soaked, pale green to dark brown lesions on leaf margins and tips",
            "White cottony sporulation (sporangiophores) visible on underside of leaves in humid conditions",
            "Lesions expand rapidly, turning black and necrotic",
            "Brown to black discoloration of stem tissue",
            "Infected tubers show copper-brown, granular rot beneath the skin",
            "Characteristic foul odor from rotting tissue",
        ],
        "favorable_conditions": [
            "Cool, wet conditions: temperatures 10–24°C (50–75°F)",
            "Relative humidity >90% or leaf wetness >12 hours",
            "Overcast, rainy weather for 2+ consecutive days",
            "Dense foliage with poor air movement",
            "Infected seed tubers carrying the pathogen",
        ],
        "economic_impact": (
            "Can cause 100% crop loss in 2–3 weeks under epidemic conditions. "
            "Major global economic impact — estimated $6.7 billion/year in losses and control costs."
        ),
        "treatments": {
            "chemical": [
                {
                    "treatment_name": "Metalaxyl + Mancozeb (e.g., Ridomil Gold)",
                    "active_ingredient": "Metalaxyl-M 4% + Mancozeb 64%",
                    "application_method": "Foliar spray",
                    "dosage": "2.5 kg/ha",
                    "frequency": "Every 7–10 days; switch to protectant after haulm destruction",
                    "waiting_period_days": 7,
                    "efficacy_score": 0.93,
                    "cost_level": "medium",
                    "availability": "common",
                    "notes": "Systemic phenylamide — critical to rotate to avoid resistance development.",
                },
                {
                    "treatment_name": "Cymoxanil + Mancozeb (e.g., Curzate)",
                    "active_ingredient": "Cymoxanil 8% + Mancozeb 64%",
                    "application_method": "Foliar spray",
                    "dosage": "2.0–2.5 kg/ha",
                    "frequency": "Every 7 days",
                    "waiting_period_days": 7,
                    "efficacy_score": 0.88,
                    "cost_level": "medium",
                    "availability": "common",
                    "notes": "Best used curatively within 72h of infection. Rotate fungicide groups.",
                },
                {
                    "treatment_name": "Fluopicolide + Propamocarb (e.g., Infinito)",
                    "active_ingredient": "Fluopicolide 6.25% + Propamocarb 62.5%",
                    "application_method": "Foliar spray",
                    "dosage": "1.6 L/ha",
                    "frequency": "Every 10–14 days; max 4 applications",
                    "waiting_period_days": 7,
                    "efficacy_score": 0.95,
                    "cost_level": "high",
                    "availability": "common",
                },
            ],
            "organic": [
                {
                    "treatment_name": "Copper hydroxide (e.g., Kocide)",
                    "active_ingredient": "Copper hydroxide",
                    "application_method": "Foliar spray, ensure thorough coverage",
                    "dosage": "2–3 kg/ha",
                    "frequency": "Every 5–7 days in wet weather",
                    "waiting_period_days": 0,
                    "efficacy_score": 0.65,
                    "cost_level": "low",
                    "availability": "common",
                    "notes": "Best as preventive treatment. Less effective once disease is established.",
                },
            ],
            "cultural": [
                {
                    "treatment_name": "Use certified disease-free seed tubers",
                    "application_method": "Source seed from certified suppliers. Inspect for tuber blight before planting.",
                    "efficacy_score": 0.85,
                    "cost_level": "medium",
                },
                {
                    "treatment_name": "Haulm destruction before harvest",
                    "application_method": "Destroy haulm 2–3 weeks before harvest to allow skin set and prevent tuber infection.",
                    "efficacy_score": 0.80,
                    "cost_level": "low",
                },
                {
                    "treatment_name": "Avoid overhead irrigation",
                    "application_method": "Use drip/furrow irrigation. If overhead irrigation is unavoidable, water in early morning.",
                    "efficacy_score": 0.70,
                    "cost_level": "medium",
                },
            ],
            "biological": [
                {
                    "treatment_name": "Bacillus amyloliquefaciens (e.g., Amylo-X)",
                    "application_method": "Foliar spray, preventive",
                    "dosage": "250–500g per 100L water",
                    "frequency": "Every 5–7 days",
                    "waiting_period_days": 0,
                    "efficacy_score": 0.52,
                    "cost_level": "medium",
                    "availability": "specialty",
                },
            ],
        },
        "preventive_practices": [
            "Plant resistant varieties (e.g., Sarpo Mira, Cara, Verity).",
            "Use a late blight forecasting service (e.g., BlightWatch, MetOffice Blight Alert).",
            "Begin preventive fungicide sprays before disease onset in high-risk periods.",
            "Ensure good soil drainage — avoid waterlogged fields.",
            "Do not plant in fields with infected volunteer plants or cull piles.",
            "Destroy all cull piles and volunteer plants before planting.",
            "Store tubers in cool, dry, well-ventilated conditions.",
        ],
        "resistant_varieties": ["Sarpo Mira", "Sarpo Axona", "Cara", "Verity", "Setanta"],
    },

    {
        "disease_code": "corn___northern_leaf_blight",
        "display_name": "Corn — Northern Leaf Blight",
        "crop_name": "corn",
        "scientific_name": "Exserohilum turcicum (Helminthosporium turcicum)",
        "pathogen_type": "fungal",
        "description": (
            "Northern leaf blight (NLB) is a common fungal disease of corn caused by "
            "Exserohilum turcicum. It is one of the most important foliar diseases of "
            "corn globally, particularly in temperate and subtropical regions. "
            "The disease is favored by moderate temperatures and extended periods of "
            "leaf wetness, and epidemics can develop rapidly under these conditions."
        ),
        "symptoms": [
            "Long, elliptical (cigar-shaped) grayish-green to tan lesions on leaves",
            "Lesions typically 2.5–15 cm (1–6 inches) long",
            "Lesions may appear water-soaked initially, then tan to grayish with wavy margins",
            "Dark olive-green to black sporulation (conidia) visible on lesion surface",
            "Severe infection causes leaf blighting and premature senescence",
            "Lower leaves usually infected first; disease moves up plant",
        ],
        "favorable_conditions": [
            "Moderate temperatures 18–27°C (65–80°F)",
            "Extended leaf wetness (>6 hours) from dew or rain",
            "High relative humidity",
            "Reduced tillage systems with corn residue on surface",
            "Susceptible corn hybrids",
        ],
        "economic_impact": (
            "Yield losses of 30–50% in susceptible hybrids under severe epidemic conditions. "
            "Losses are greatest when infection occurs before or during silking."
        ),
        "treatments": {
            "chemical": [
                {
                    "treatment_name": "Propiconazole (e.g., Tilt)",
                    "active_ingredient": "Propiconazole",
                    "application_method": "Aerial or ground foliar spray",
                    "dosage": "0.5 L/ha",
                    "frequency": "At first sign; repeat in 14 days if needed",
                    "waiting_period_days": 30,
                    "efficacy_score": 0.82,
                    "cost_level": "low",
                    "availability": "common",
                },
                {
                    "treatment_name": "Azoxystrobin + Propiconazole (e.g., Quilt Xcel)",
                    "active_ingredient": "Azoxystrobin 13.5% + Propiconazole 11.7%",
                    "application_method": "Foliar spray at VT/R1 growth stage",
                    "dosage": "1.0–1.5 L/ha",
                    "frequency": "Single application at tassel/silk timing for best ROI",
                    "waiting_period_days": 30,
                    "efficacy_score": 0.88,
                    "cost_level": "medium",
                    "availability": "common",
                    "notes": "Most economical to apply at tassel emergence (VT) when disease pressure is high.",
                },
            ],
            "organic": [
                {
                    "treatment_name": "Copper-based fungicide",
                    "application_method": "Foliar spray at early infection",
                    "dosage": "2–3 kg/ha copper hydroxide",
                    "frequency": "Every 10 days",
                    "waiting_period_days": 0,
                    "efficacy_score": 0.50,
                    "cost_level": "low",
                },
            ],
            "cultural": [
                {
                    "treatment_name": "Plant resistant hybrids",
                    "application_method": "Select hybrids with Ht1, Ht2, HtN, or Htm1 resistance genes from seed catalog ratings.",
                    "efficacy_score": 0.90,
                    "cost_level": "low",
                },
                {
                    "treatment_name": "Crop rotation",
                    "application_method": "Rotate out of corn for 1 year to reduce inoculum from infected residue.",
                    "efficacy_score": 0.70,
                    "cost_level": "low",
                },
                {
                    "treatment_name": "Tillage",
                    "application_method": "Bury or incorporate corn residue through tillage to accelerate decomposition and reduce inoculum.",
                    "efficacy_score": 0.60,
                    "cost_level": "medium",
                },
            ],
            "biological": [],
        },
        "preventive_practices": [
            "Select resistant or tolerant hybrids — most effective long-term strategy.",
            "Scout fields weekly from V6 through R3, especially after wet periods.",
            "Apply foliar fungicide preventively at VT/R1 in high-risk years.",
            "Manage crop residue with tillage or cover crops.",
            "Maintain optimum plant population — avoid excessive density that reduces airflow.",
        ],
        "resistant_varieties": [
            "Hybrids carrying Ht1, Ht2, HtN resistance genes (consult seed catalog)"
        ],
    },
]

# Healthy class document (shared pattern)
HEALTHY_CLASSES = [
    {"crop": "tomato", "code": "tomato___healthy"},
    {"crop": "potato", "code": "potato___healthy"},
    {"crop": "corn", "code": "corn___healthy"},
    {"crop": "apple", "code": "apple___healthy"},
    {"crop": "grape", "code": "grape___healthy"},
    {"crop": "pepper", "code": "pepper___healthy"},
    {"crop": "peach", "code": "peach___healthy"},
    {"crop": "strawberry", "code": "strawberry___healthy"},
    {"crop": "cherry", "code": "cherry___healthy"},
    {"crop": "blueberry", "code": "blueberry___healthy"},
    {"crop": "raspberry", "code": "raspberry___healthy"},
    {"crop": "soybean", "code": "soybean___healthy"},
]

for h in HEALTHY_CLASSES:
    DISEASE_DOCUMENTS.append({
        "disease_code": h["code"],
        "display_name": f"{h['crop'].title()} — Healthy",
        "crop_name": h["crop"],
        "scientific_name": None,
        "pathogen_type": None,
        "description": f"No disease detected. The {h['crop']} plant appears healthy.",
        "symptoms": [],
        "favorable_conditions": [],
        "economic_impact": "None",
        "treatments": {},
        "preventive_practices": [
            "Continue regular monitoring.",
            "Maintain balanced nutrition.",
            "Practice crop rotation.",
            "Use disease-resistant varieties.",
            "Ensure good soil drainage.",
        ],
    })


async def seed():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db.diseases

    # Create indexes
    await collection.create_index("disease_code", unique=True)
    await collection.create_index("crop_name")

    inserted = 0
    updated = 0
    for doc in DISEASE_DOCUMENTS:
        result = await collection.update_one(
            {"disease_code": doc["disease_code"]},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        else:
            updated += 1

    logger.info(f"Seed complete: {inserted} inserted, {updated} updated")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
