// MongoDB initialization — runs once at container start
// Creates indexes on the diseases collection for fast lookups.
// Full disease data is seeded by scripts/seed_disease_kb.py

db = db.getSiblingDB('plant_disease_kb');

// Create indexes
db.diseases.createIndex({ disease_code: 1 }, { unique: true });
db.diseases.createIndex({ crop_name: 1 });
db.diseases.createIndex({ pathogen_type: 1 });
db.diseases.createIndex(
    { display_name: "text", description: "text" },
    { name: "diseases_text_search" }
);

print("MongoDB indexes created on plant_disease_kb.diseases");
