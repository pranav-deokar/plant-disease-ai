/**
 * Admin Dashboard
 * System statistics, model management, prediction monitoring.
 * Only accessible with role=admin.
 */

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const api = axios.create({ baseURL: API_BASE, withCredentials: true });

// ── Data fetchers ──────────────────────────────────────────────────────────────
const fetchStats = () => api.get("/api/v1/admin/stats").then((r) => r.data);
const fetchModels = () => api.get("/api/v1/admin/models").then((r) => r.data);
const triggerRetrain = () => api.post("/api/v1/admin/retrain").then((r) => r.data);
const activateModel = (id) => api.post(`/api/v1/admin/models/${id}/activate`).then((r) => r.data);

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = "blue", icon }) {
  const colorMap = {
    blue:   "bg-blue-50 border-blue-100",
    green:  "bg-green-50 border-green-100",
    amber:  "bg-amber-50 border-amber-100",
    purple: "bg-purple-50 border-purple-100",
    red:    "bg-red-50 border-red-100",
  };
  const textMap = {
    blue:   "text-blue-700",
    green:  "text-green-700",
    amber:  "text-amber-700",
    purple: "text-purple-700",
    red:    "text-red-700",
  };
  return (
    <div className={`rounded-xl border p-5 ${colorMap[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500 mb-1">{label}</p>
          <p className={`text-3xl font-bold ${textMap[color]}`}>{value}</p>
          {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
        </div>
        {icon && <span className="text-2xl opacity-60">{icon}</span>}
      </div>
    </div>
  );
}

// ── Model Row ─────────────────────────────────────────────────────────────────
function ModelRow({ model, onActivate }) {
  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      <td className="px-4 py-3">
        <div className="font-medium text-sm text-gray-900">{model.model_name}</div>
        <div className="text-xs text-gray-400">{model.architecture}</div>
      </td>
      <td className="px-4 py-3 text-sm text-gray-700">{model.version}</td>
      <td className="px-4 py-3 text-sm text-gray-700">
        {model.val_accuracy ? `${(model.val_accuracy * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="px-4 py-3 text-sm text-gray-700">
        {model.val_f1_macro ? `${(model.val_f1_macro * 100).toFixed(1)}%` : "—"}
      </td>
      <td className="px-4 py-3 text-sm text-gray-700">{model.total_predictions?.toLocaleString() || "—"}</td>
      <td className="px-4 py-3">
        {model.is_active ? (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            Active
          </span>
        ) : model.is_shadow ? (
          <span className="px-2.5 py-1 bg-purple-100 text-purple-700 rounded-full text-xs">Shadow</span>
        ) : (
          <span className="px-2.5 py-1 bg-gray-100 text-gray-500 rounded-full text-xs">Inactive</span>
        )}
      </td>
      <td className="px-4 py-3">
        {!model.is_active && (
          <button
            onClick={() => onActivate(model.id)}
            className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
          >
            Activate
          </button>
        )}
      </td>
    </tr>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────
export default function AdminPage() {
  const queryClient = useQueryClient();
  const [retrainStatus, setRetrainStatus] = useState(null);

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["admin-stats"],
    queryFn: fetchStats,
    refetchInterval: 30_000,
  });

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ["admin-models"],
    queryFn: fetchModels,
  });

  const retrainMutation = useMutation({
    mutationFn: triggerRetrain,
    onSuccess: (data) => setRetrainStatus(`✓ ${data.message}`),
    onError: () => setRetrainStatus("✗ Retraining trigger failed."),
  });

  const activateMutation = useMutation({
    mutationFn: activateModel,
    onSuccess: () => queryClient.invalidateQueries(["admin-models"]),
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🌿</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Admin Dashboard</h1>
            <p className="text-xs text-gray-400">Plant Disease AI — System Overview</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => queryClient.invalidateQueries()}
            className="px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-600 hover:bg-gray-50"
          >
            ↻ Refresh
          </button>
          <button
            onClick={() => retrainMutation.mutate()}
            disabled={retrainMutation.isPending}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
          >
            {retrainMutation.isPending ? "Submitting..." : "⚡ Trigger Retraining"}
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {retrainStatus && (
          <div className={`px-4 py-3 rounded-lg text-sm ${retrainStatus.startsWith("✓") ? "bg-green-50 text-green-700 border border-green-200" : "bg-red-50 text-red-700 border border-red-200"}`}>
            {retrainStatus}
          </div>
        )}

        {/* Stat Cards */}
        {statsLoading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="bg-gray-100 rounded-xl h-28 animate-pulse" />
            ))}
          </div>
        ) : stats ? (
          <>
            <section>
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">Prediction Volume</h2>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard label="Total Predictions" value={stats.total_predictions?.toLocaleString()} icon="🔬" color="blue" />
                <StatCard label="Today" value={stats.predictions_today?.toLocaleString()} sub="Last 24 hours" icon="📅" color="green" />
                <StatCard label="This Week" value={stats.predictions_this_week?.toLocaleString()} sub="Last 7 days" icon="📊" color="purple" />
                <StatCard label="Unique Users Today" value={stats.unique_users_today?.toLocaleString()} icon="👩‍🌾" color="amber" />
              </div>
            </section>

            <section>
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">Model Performance</h2>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                <StatCard
                  label="Avg Confidence"
                  value={`${(stats.avg_confidence * 100).toFixed(1)}%`}
                  sub="Across all predictions"
                  icon="🎯"
                  color="green"
                />
                <StatCard
                  label="Avg Processing Time"
                  value={`${Math.round(stats.avg_processing_ms)}ms`}
                  sub="End-to-end inference"
                  icon="⚡"
                  color="blue"
                />
                <StatCard
                  label="Feedback Accuracy"
                  value={`${(stats.correct_feedback_rate * 100).toFixed(1)}%`}
                  sub="User-confirmed correct predictions"
                  icon="✓"
                  color={stats.correct_feedback_rate > 0.9 ? "green" : "amber"}
                />
              </div>
            </section>

            {/* Top Diseases */}
            {stats.top_diseases?.length > 0 && (
              <section>
                <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">Top Detected Diseases</h2>
                <div className="bg-white rounded-xl border border-gray-200 p-5">
                  <div className="space-y-3">
                    {stats.top_diseases.slice(0, 8).map((d, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-xs text-gray-400 w-4">{i + 1}</span>
                        <div className="flex-1">
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-700 font-medium">{d.disease_name}</span>
                            <span className="text-gray-500">{d.count.toLocaleString()}</span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-1.5">
                            <div
                              className="bg-green-500 h-1.5 rounded-full"
                              style={{ width: `${(d.count / stats.top_diseases[0].count) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}
          </>
        ) : null}

        {/* Model Registry */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">Model Registry</h2>
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            {modelsLoading ? (
              <div className="p-8 text-center text-gray-400">Loading models...</div>
            ) : (
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200 text-xs text-gray-500 uppercase tracking-wider">
                    <th className="px-4 py-3">Model</th>
                    <th className="px-4 py-3">Version</th>
                    <th className="px-4 py-3">Val Acc</th>
                    <th className="px-4 py-3">F1 Macro</th>
                    <th className="px-4 py-3">Predictions</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(models || []).map((m) => (
                    <ModelRow key={m.id} model={m} onActivate={(id) => activateMutation.mutate(id)} />
                  ))}
                </tbody>
              </table>
            )}
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Active models serve production traffic. Shadow models run in parallel for A/B testing.
          </p>
        </section>

        {/* Active Model Versions */}
        {stats?.model_versions_active?.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Currently Active Models</h3>
            <div className="flex flex-wrap gap-2">
              {stats.model_versions_active.map((name) => (
                <span key={name} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-mono">
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
