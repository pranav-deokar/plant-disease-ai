/**
 * Disease Database Browser
 * Browse, search, and read full disease profiles with treatment details.
 */

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const PATHOGEN_COLORS = {
  fungal:      "bg-orange-100 text-orange-700",
  bacterial:   "bg-blue-100 text-blue-700",
  viral:       "bg-purple-100 text-purple-700",
  oomycete:    "bg-red-100 text-red-700",
  pest:        "bg-yellow-100 text-yellow-700",
  nutritional: "bg-gray-100 text-gray-600",
};

const IMPACT_COLORS = {
  low:      "text-green-600",
  medium:   "text-yellow-600",
  high:     "text-orange-600",
  critical: "text-red-600 font-bold",
};

async function fetchDiseases({ crop, pathogen, search, skip, limit }) {
  const params = new URLSearchParams({ skip, limit, include_healthy: "false" });
  if (crop) params.append("crop_name", crop);
  if (pathogen) params.append("pathogen_type", pathogen);
  const { data } = await axios.get(`${API_BASE}/api/v1/diseases/?${params}`);
  return data;
}

async function fetchCrops() {
  const { data } = await axios.get(`${API_BASE}/api/v1/diseases/crops`);
  return data.crops;
}

async function fetchDiseaseDetail(code) {
  const { data } = await axios.get(`${API_BASE}/api/v1/diseases/${code}`);
  return data;
}

function DiseaseCard({ disease, onClick }) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-white rounded-xl border border-gray-200 hover:border-green-300 hover:shadow-md transition-all p-5 group"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 group-hover:text-green-700 transition-colors truncate">
            {disease.display_name}
          </h3>
          <p className="text-sm text-gray-400 mt-0.5 font-mono truncate">{disease.disease_code}</p>
        </div>
        <span className={`flex-shrink-0 px-2 py-1 rounded-full text-xs ${PATHOGEN_COLORS[disease.pathogen_type] || "bg-gray-100 text-gray-500"}`}>
          {disease.pathogen_type || "unknown"}
        </span>
      </div>

      <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
        {disease.model_accuracy && (
          <span>Model acc: <strong>{(disease.model_accuracy * 100).toFixed(1)}%</strong></span>
        )}
        {disease.economic_impact && (
          <span className={IMPACT_COLORS[disease.economic_impact] || ""}>
            Impact: {disease.economic_impact}
          </span>
        )}
        {disease.is_contagious && (
          <span className="text-red-400">⚠ Contagious</span>
        )}
      </div>
    </button>
  );
}

function DetailPanel({ code, onClose }) {
  const { data, isLoading } = useQuery({
    queryKey: ["disease-detail", code],
    queryFn: () => fetchDiseaseDetail(code),
    enabled: !!code,
  });

  const TREATMENT_ICONS = { chemical: "⚗️", organic: "🌿", cultural: "🌾", biological: "🦠" };

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex justify-end" onClick={onClose}>
      <div
        className="w-full max-w-xl bg-white h-full overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
          <h2 className="font-bold text-lg text-gray-900">Disease Details</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 text-2xl leading-none">×</button>
        </div>

        {isLoading ? (
          <div className="p-12 text-center text-gray-400">
            <div className="text-4xl mb-3 animate-pulse">🔬</div>
            <p>Loading...</p>
          </div>
        ) : data ? (
          <div className="p-6 space-y-6">
            {/* Header */}
            <div>
              <h3 className="text-2xl font-bold text-gray-900">{data.display_name}</h3>
              {data.scientific_name && (
                <p className="text-sm text-gray-500 italic mt-0.5">{data.scientific_name}</p>
              )}
              <div className="flex flex-wrap gap-2 mt-3">
                {data.pathogen_type && (
                  <span className={`px-3 py-1 rounded-full text-xs ${PATHOGEN_COLORS[data.pathogen_type] || "bg-gray-100"}`}>
                    {data.pathogen_type}
                  </span>
                )}
                {data.economic_impact && (
                  <span className={`px-3 py-1 rounded-full text-xs bg-gray-100 ${IMPACT_COLORS[data.economic_impact] || ""}`}>
                    {data.economic_impact} economic impact
                  </span>
                )}
              </div>
            </div>

            {/* Description */}
            {data.knowledge?.description && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">About</h4>
                <p className="text-sm text-gray-600 leading-relaxed">{data.knowledge.description}</p>
              </div>
            )}

            {/* Symptoms */}
            {data.knowledge?.symptoms?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Symptoms</h4>
                <ul className="space-y-1.5">
                  {data.knowledge.symptoms.map((s, i) => (
                    <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="text-orange-400 mt-0.5">•</span>
                      <span>{s}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Favorable conditions */}
            {data.knowledge?.favorable_conditions?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Favorable Conditions</h4>
                <ul className="space-y-1.5">
                  {data.knowledge.favorable_conditions.map((c, i) => (
                    <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="text-blue-400 mt-0.5">→</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Treatments */}
            {data.treatments?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-3">Treatments</h4>
                {["chemical", "organic", "cultural", "biological"].map((type) => {
                  const items = data.treatments.filter((t) => t.treatment_type === type);
                  if (!items.length) return null;
                  return (
                    <div key={type} className="mb-4">
                      <h5 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                        {TREATMENT_ICONS[type]} {type}
                      </h5>
                      <div className="space-y-2">
                        {items.map((t, i) => (
                          <div key={i} className="bg-gray-50 rounded-lg p-3 text-sm">
                            <div className="font-medium text-gray-800">{t.treatment_name}</div>
                            {t.application_method && (
                              <div className="text-gray-500 text-xs mt-1">{t.application_method}</div>
                            )}
                            {t.dosage && <div className="text-gray-400 text-xs mt-0.5">Dosage: {t.dosage}</div>}
                            {t.waiting_period_days && (
                              <div className="text-gray-400 text-xs mt-0.5">
                                Pre-harvest interval: {t.waiting_period_days} days
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Preventive practices */}
            {data.knowledge?.preventive_practices?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Preventive Practices</h4>
                <ul className="space-y-1.5">
                  {data.knowledge.preventive_practices.map((p, i) => (
                    <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="text-green-500 mt-0.5">✓</span>
                      <span>{p}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Model metadata */}
            <div className="text-xs text-gray-400 border-t pt-4 space-y-1">
              <div>Disease code: <span className="font-mono">{data.disease_code}</span></div>
              <div>Class index: {data.class_index} · Training samples: {data.training_samples?.toLocaleString()}</div>
              {data.model_accuracy && <div>Per-class model accuracy: {(data.model_accuracy * 100).toFixed(1)}%</div>}
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-400">No data available.</div>
        )}
      </div>
    </div>
  );
}

export default function DiseasesPage() {
  const [crop, setCrop] = useState("");
  const [pathogen, setPathogen] = useState("");
  const [selectedCode, setSelectedCode] = useState(null);
  const [search, setSearch] = useState("");

  const { data: crops } = useQuery({ queryKey: ["crops"], queryFn: fetchCrops });
  const { data: diseases, isLoading } = useQuery({
    queryKey: ["diseases", crop, pathogen],
    queryFn: () => fetchDiseases({ crop, pathogen, skip: 0, limit: 200 }),
  });

  const filtered = (diseases || []).filter((d) =>
    !search || d.display_name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4 flex-wrap">
        <Link to="/" className="text-gray-400 hover:text-gray-700 text-sm">← Back</Link>
        <h1 className="text-xl font-bold text-gray-900">Disease Database</h1>
        <span className="text-sm text-gray-400">
          {filtered.length} {filtered.length === 1 ? "disease" : "diseases"}
        </span>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Filters */}
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6 flex flex-wrap gap-3 items-center">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search diseases..."
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm flex-1 min-w-40"
          />
          <select
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All Crops</option>
            {(crops || []).map((c) => (
              <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
            ))}
          </select>
          <select
            value={pathogen}
            onChange={(e) => setPathogen(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All Pathogens</option>
            {["fungal", "bacterial", "viral", "oomycete", "pest", "nutritional"].map((p) => (
              <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
            ))}
          </select>
        </div>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(9)].map((_, i) => (
              <div key={i} className="bg-gray-100 rounded-xl h-28 animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((d) => (
              <DiseaseCard key={d.disease_code} disease={d} onClick={() => setSelectedCode(d.disease_code)} />
            ))}
          </div>
        )}
      </main>

      {selectedCode && (
        <DetailPanel code={selectedCode} onClose={() => setSelectedCode(null)} />
      )}
    </div>
  );
}
