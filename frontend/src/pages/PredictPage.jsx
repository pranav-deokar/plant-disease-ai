import React, { useCallback, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const predictDisease = async ({ imageFile, cropHint, topK, recommendationMode }) => {
  const formData = new FormData();
  formData.append("image", imageFile);
  if (cropHint) formData.append("crop_hint", cropHint);
  formData.append("top_k", topK.toString());
  formData.append("recommendation_mode", recommendationMode);

  const response = await axios.post(`${API_BASE}/api/v1/predictions/`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    withCredentials: true,
  });

  return response.data;
};

const SEVERITY_COLORS = {
  healthy: "bg-green-100 text-green-800 border-green-200",
  mild: "bg-yellow-100 text-yellow-800 border-yellow-200",
  moderate: "bg-orange-100 text-orange-800 border-orange-200",
  severe: "bg-red-100 text-red-800 border-red-200",
  critical: "bg-red-200 text-red-900 border-red-300 animate-pulse",
};

const SeverityBadge = ({ severity = "unknown" }) => (
  <span
    className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium ${
      SEVERITY_COLORS[severity] || "bg-gray-100 text-gray-700 border-gray-200"
    }`}
  >
    {severity.charAt(0).toUpperCase() + severity.slice(1)}
  </span>
);

const ConfidenceBar = ({ confidence, label }) => (
  <div className="grid grid-cols-[minmax(0,1fr)_72px] items-center gap-3">
    <div className="min-w-0">
      <div className="mb-1 truncate text-sm text-gray-600">{label}</div>
      <div className="h-2 rounded-full bg-gray-100">
        <div
          className="h-2 rounded-full bg-green-500 transition-all duration-500"
          style={{ width: `${(confidence * 100).toFixed(1)}%` }}
        />
      </div>
    </div>
    <span className="text-right text-sm font-semibold text-gray-700">
      {(confidence * 100).toFixed(1)}%
    </span>
  </div>
);

const TreatmentCard = ({ type, treatments }) => {
  const icons = {
    chemical: "Shield",
    organic: "Leaf",
    cultural: "Field",
    biological: "Bio",
  };

  const colors = {
    chemical: "border-blue-200 bg-blue-50/80",
    organic: "border-green-200 bg-green-50/80",
    cultural: "border-amber-200 bg-amber-50/80",
    biological: "border-violet-200 bg-violet-50/80",
  };

  return (
    <div className={`h-full rounded-2xl border p-5 shadow-sm ${colors[type] || "border-gray-200 bg-gray-50"}`}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h4 className="text-base font-semibold text-gray-800">
          {type.charAt(0).toUpperCase() + type.slice(1)} Treatment
        </h4>
        <span className="rounded-full bg-white/80 px-2.5 py-1 text-xs font-medium text-gray-600">
          {icons[type] || "Guide"}
        </span>
      </div>

      <div className="space-y-3">
        {treatments.map((t, i) => (
          <div key={i} className="rounded-xl border border-white/90 bg-white/90 p-4 shadow-sm">
            <div className="text-sm font-semibold text-gray-800">{t.treatment_name}</div>
            {t.application_method && (
              <div className="mt-2 text-sm leading-relaxed text-gray-600">{t.application_method}</div>
            )}
            {t.dosage && (
              <div className="mt-3 inline-flex rounded-full bg-gray-100 px-2.5 py-1 text-xs text-gray-500">
                Dosage: {t.dosage}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

const ResultPanel = ({ result, imagePreview }) => {
  const [showGradCAM, setShowGradCAM] = useState(false);
  const isHealthy = result.disease_code?.endsWith("healthy");
  const treatmentsByType = result.treatments?.treatments || {};

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      <div
        className={`rounded-2xl border-2 p-6 shadow-sm xl:col-span-12 ${
          isHealthy ? "border-green-300 bg-green-50" : "border-red-200 bg-red-50"
        }`}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="mb-1 text-sm text-gray-500">Detected condition</p>
            <h2 className="text-2xl font-bold text-gray-900">{result.disease_name}</h2>
            <p className="mt-1 text-sm text-gray-500">
              {isHealthy ? "No disease detected, plant appears healthy." : `Disease code: ${result.disease_code}`}
            </p>
          </div>
          <SeverityBadge severity={result.severity} />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-gray-600">
          <span>
            Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
          </span>
          <span>
            Severity: <strong>{(result.severity_score * 100).toFixed(0)}/100</strong>
          </span>
          <span>
            Time: <strong>{result.processing_ms}ms</strong>
          </span>
        </div>
      </div>

      {result.gradcam_url && (
        <div className="xl:col-span-7">
          <div className="mb-3 flex items-center gap-3">
            <h3 className="font-bold text-gray-900">Disease Region Map</h3>
            <button
              onClick={() => setShowGradCAM(!showGradCAM)}
              className="text-sm font-medium text-blue-600 underline"
            >
              {showGradCAM ? "Show original" : "Show Grad-CAM"}
            </button>
          </div>

          <div className="overflow-hidden rounded-2xl border bg-white shadow-sm">
            <img
              src={showGradCAM ? result.gradcam_url : imagePreview}
              alt="Prediction preview"
              className="h-[320px] w-full bg-slate-50 object-contain"
            />
          </div>
        </div>
      )}

      <div
        className={`rounded-2xl border border-gray-200 bg-white p-5 shadow-sm ${
          result.gradcam_url ? "xl:col-span-5" : "xl:col-span-12"
        }`}
      >
        <h3 className="mb-4 font-bold text-gray-900">Top Predictions</h3>
        <div className="space-y-3">
          {result.top_k?.map((p) => (
            <ConfidenceBar
              key={p.rank}
              label={`${p.rank}. ${p.disease_name}`}
              confidence={p.confidence}
            />
          ))}
        </div>
      </div>

      {!isHealthy && Object.keys(treatmentsByType).length > 0 && (
        <div className="xl:col-span-12">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-bold text-gray-900">Treatment Options</h3>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Object.entries(treatmentsByType).map(([type, items]) => (
              <TreatmentCard key={type} type={type} treatments={items} />
            ))}
          </div>
        </div>
      )}

      {result.treatments && (
        <div className="mt-2 space-y-4 xl:col-span-12">
          <span className="inline-flex rounded-full bg-blue-100 px-2 py-1 text-xs text-blue-700">
            AI Generated
          </span>

          {result.treatments.description && (
            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <h3 className="mb-2 font-bold">About Disease</h3>
              <p className="text-sm leading-relaxed text-gray-600">{result.treatments.description}</p>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {result.treatments.symptoms?.length > 0 && (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4">
                <h3 className="mb-2 font-semibold text-red-800">Symptoms</h3>
                <ul className="space-y-1 text-sm text-gray-700">
                  {result.treatments.symptoms.map((s, i) => (
                    <li key={i}>- {s}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.treatments.favorable_conditions?.length > 0 && (
              <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
                <h3 className="mb-2 font-semibold text-blue-800">Conditions</h3>
                <ul className="space-y-1 text-sm text-gray-700">
                  {result.treatments.favorable_conditions.map((c, i) => (
                    <li key={i}>- {c}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.treatments.preventive_practices?.length > 0 && (
              <div className="rounded-xl border border-green-200 bg-green-50 p-4 md:col-span-2">
                <h3 className="mb-2 font-semibold text-green-800">Prevention</h3>
                <ul className="space-y-1 text-sm text-gray-700">
                  {result.treatments.preventive_practices.map((p, i) => (
                    <li key={i}>- {p}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="border-t pt-3 text-xs text-gray-400 xl:col-span-12">
        Prediction ID: {result.prediction_id}
      </div>
    </div>
  );
};

export default function PredictPage() {
  const queryClient = useQueryClient();
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [cropHint, setCropHint] = useState("");
  const [topK, setTopK] = useState(5);
  const [recommendationMode, setRecommendationMode] = useState("db");

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles?.[0];
    if (!file) return;

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [],
      "image/png": [],
      "image/webp": [],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024,
  });

  const mutation = useMutation({
    mutationFn: predictDisease,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });

  const handleSubmit = () => {
    if (!imageFile) return;

    mutation.mutate({
      imageFile,
      cropHint,
      topK,
      recommendationMode,
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-emerald-50">
      <header className="border-b border-gray-200 bg-white shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🌿</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Plant Disease AI</h1>
              <p className="text-xs text-gray-500">Detect and treat crop diseases instantly</p>
            </div>
          </div>
          <nav className="flex gap-6 text-sm text-gray-600">
            <a href="/history" className="hover:text-green-700">History</a>
            <a href="/diseases" className="hover:text-green-700">Disease DB</a>
            <a href="/dashboard" className="hover:text-green-700">Dashboard</a>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-10">
        <div className="grid grid-cols-1 items-start gap-10 xl:grid-cols-12">
          <div className="space-y-4 xl:col-span-4 xl:sticky xl:top-8">
            <div>
              <h2 className="text-3xl font-bold text-gray-900">Diagnose Your Plant</h2>
              <p className="mt-2 text-gray-500">
                Upload a clear photo of a leaf. Our AI will identify the disease and suggest treatment.
              </p>
            </div>

            <div
              {...getRootProps()}
              className={`rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-all ${
                isDragActive
                  ? "border-green-400 bg-green-50 scale-[1.01]"
                  : "border-gray-300 hover:border-green-400 hover:bg-green-50"
              }`}
            >
              <input {...getInputProps()} />
              {imagePreview ? (
                <div className="space-y-3">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="mx-auto max-h-56 rounded-xl object-contain"
                  />
                  <p className="text-sm text-gray-500">{imageFile?.name} . Click to change</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="text-5xl">Leaf</div>
                  <p className="font-medium text-gray-600">
                    {isDragActive ? "Drop the image here..." : "Drag and drop a leaf photo here"}
                  </p>
                  <p className="text-sm text-gray-400">or click to browse . JPEG, PNG, WebP . max 10 MB</p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-sm font-medium text-gray-700">Crop type (optional)</label>
                <select
                  value={cropHint}
                  onChange={(e) => setCropHint(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-400"
                >
                  <option value="">Auto-detect</option>
                  {["tomato", "potato", "corn", "apple", "grape", "pepper", "peach", "orange", "strawberry", "cherry"].map((c) => (
                    <option key={c} value={c}>
                      {c.charAt(0).toUpperCase() + c.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-sm font-medium text-gray-700">Alternatives to show</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-400"
                >
                  {[3, 5, 10].map((n) => (
                    <option key={n} value={n}>
                      Top {n}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => setRecommendationMode("db")}
                className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                  recommendationMode === "db"
                    ? "border-green-600 bg-green-600 text-white"
                    : "bg-white text-gray-700 hover:border-green-300"
                }`}
              >
                Verified
              </button>

              <button
                type="button"
                onClick={() => setRecommendationMode("ai")}
                className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                  recommendationMode === "ai"
                    ? "border-blue-600 bg-blue-600 text-white"
                    : "bg-white text-gray-700 hover:border-blue-300"
                }`}
              >
                AI Recommendations
              </button>
            </div>

            <button
              onClick={handleSubmit}
              disabled={!imageFile || mutation.isPending}
              className={`w-full rounded-2xl py-4 text-lg font-semibold transition-all ${
                !imageFile || mutation.isPending
                  ? "cursor-not-allowed bg-gray-200 text-gray-400"
                  : "bg-green-600 text-white shadow-lg shadow-green-200 hover:bg-green-700 active:scale-[0.98]"
              }`}
            >
              {mutation.isPending ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Analyzing leaf...
                </span>
              ) : (
                "Detect Disease"
              )}
            </button>

            {mutation.isError && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {mutation.error?.response?.data?.detail || "Analysis failed. Please try again."}
              </div>
            )}
          </div>

          <div className="min-w-0 xl:col-span-8">
            {mutation.isSuccess ? (
              <ResultPanel result={mutation.data} imagePreview={imagePreview} />
            ) : (
              <div className="flex min-h-[520px] items-center justify-center rounded-3xl border border-dashed border-green-200 bg-white/70 shadow-sm">
                <div className="max-w-md space-y-4 px-6 text-center text-gray-400">
                  <div className="text-6xl opacity-30">Scan</div>
                  <p className="text-lg text-gray-500">Upload a leaf image to get started</p>
                  <p className="text-sm">
                    Results will fill this panel with disease details, confidence, severity, visual analysis, and treatment recommendations.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
