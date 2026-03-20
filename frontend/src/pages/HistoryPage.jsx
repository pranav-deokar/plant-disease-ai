/**
 * Prediction History Page
 * Paginated list of past predictions with filter/search, exportable.
 */

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const SEVERITY_COLORS = {
  healthy:  "bg-green-100 text-green-800",
  mild:     "bg-yellow-100 text-yellow-700",
  moderate: "bg-orange-100 text-orange-700",
  severe:   "bg-red-100 text-red-700",
  critical: "bg-red-200 text-red-900 font-bold",
};

const CROPS = ["", "tomato", "potato", "corn", "apple", "grape", "pepper", "peach", "strawberry"];

async function fetchHistory({ page, pageSize, severity, crop }) {
  const params = new URLSearchParams({ page, page_size: pageSize });
  if (severity) params.append("severity", severity);
  const { data } = await axios.get(`${API_BASE}/api/v1/predictions/?${params}`, {
    withCredentials: true,
  });
  return data;
}

function HistoryRow({ pred, onFeedback }) {
  const date = new Date(pred.created_at).toLocaleString();
  const severity = pred.severity || "unknown";

  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
      <td className="px-4 py-3">
        {pred.gradcam_url ? (
          <img
            src={pred.image_url || pred.gradcam_url}
            alt="leaf"
            className="w-12 h-12 rounded-lg object-cover border border-gray-200"
          />
        ) : (
          <div className="w-12 h-12 rounded-lg bg-gray-100 flex items-center justify-center text-xl">🍃</div>
        )}
      </td>
      <td className="px-4 py-3">
        <div className="font-medium text-gray-900 text-sm">{pred.disease_name}</div>
        <div className="text-xs text-gray-400 mt-0.5 font-mono">{pred.disease_code}</div>
      </td>
      <td className="px-4 py-3">
        <span className={`px-2 py-1 rounded-full text-xs ${SEVERITY_COLORS[severity]}`}>
          {severity}
        </span>
      </td>
      <td className="px-4 py-3 text-sm text-gray-700">
        {(pred.confidence * 100).toFixed(1)}%
      </td>
      <td className="px-4 py-3 text-xs text-gray-500">{date}</td>
      <td className="px-4 py-3 text-sm text-gray-600">{pred.processing_ms}ms</td>
      <td className="px-4 py-3">
        <div className="flex gap-2">
          <Link
            to={`/predictions/${pred.prediction_id}`}
            className="text-xs text-blue-600 hover:underline"
          >
            View
          </Link>
          {!pred.feedback && (
            <button
              onClick={() => onFeedback(pred)}
              className="text-xs text-gray-500 hover:text-green-600 hover:underline"
            >
              Feedback
            </button>
          )}
          {pred.feedback && (
            <span className={`text-xs ${pred.feedback.is_correct ? "text-green-600" : "text-red-500"}`}>
              {pred.feedback.is_correct ? "✓ Correct" : "✗ Wrong"}
            </span>
          )}
        </div>
      </td>
    </tr>
  );
}

function FeedbackModal({ prediction, onClose, onSubmit }) {
  const [isCorrect, setIsCorrect] = useState(true);
  const [correctCode, setCorrectCode] = useState("");
  const [notes, setNotes] = useState("");
  const [helpful, setHelpful] = useState(null);

  const handleSubmit = () => {
    onSubmit({
      prediction_id: prediction.prediction_id,
      is_correct: isCorrect,
      correct_disease_code: !isCorrect ? correctCode : undefined,
      user_notes: notes || undefined,
      treatment_helpful: helpful,
    });
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6 space-y-4">
        <h3 className="text-lg font-bold text-gray-900">Prediction Feedback</h3>
        <p className="text-sm text-gray-500">
          Was the prediction <strong>{prediction.disease_name}</strong> correct?
        </p>

        <div className="flex gap-3">
          {[true, false].map((val) => (
            <button
              key={String(val)}
              onClick={() => setIsCorrect(val)}
              className={`flex-1 py-2.5 rounded-xl text-sm font-medium border transition-all ${
                isCorrect === val
                  ? val ? "bg-green-500 text-white border-green-500" : "bg-red-500 text-white border-red-500"
                  : "border-gray-300 text-gray-600 hover:border-gray-400"
              }`}
            >
              {val ? "✓ Yes, correct" : "✗ No, wrong"}
            </button>
          ))}
        </div>

        {!isCorrect && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Correct disease code</label>
            <input
              type="text"
              value={correctCode}
              onChange={(e) => setCorrectCode(e.target.value)}
              placeholder="e.g. tomato___late_blight"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Were the treatments helpful?</label>
          <div className="flex gap-3">
            {[true, false, null].map((val) => (
              <button
                key={String(val)}
                onClick={() => setHelpful(val)}
                className={`px-4 py-2 rounded-lg text-sm border transition-all ${
                  helpful === val ? "bg-blue-500 text-white border-blue-500" : "border-gray-300 text-gray-600"
                }`}
              >
                {val === true ? "Yes" : val === false ? "No" : "N/A"}
              </button>
            ))}
          </div>
        </div>

        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Any additional notes..."
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none h-24"
        />

        <div className="flex gap-3 pt-2">
          <button onClick={onClose} className="flex-1 py-2.5 border border-gray-300 rounded-xl text-sm text-gray-600 hover:bg-gray-50">
            Cancel
          </button>
          <button onClick={handleSubmit} className="flex-1 py-2.5 bg-green-600 text-white rounded-xl text-sm font-medium hover:bg-green-700">
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}

export default function HistoryPage() {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [severity, setSeverity] = useState("");
  const [feedbackPred, setFeedbackPred] = useState(null);

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["predictions", page, pageSize, severity],
    queryFn: () => fetchHistory({ page, pageSize, severity, crop: "" }),
    keepPreviousData: true,
  });

  const handleFeedbackSubmit = async (payload) => {
    try {
      await axios.post(
        `${API_BASE}/api/v1/predictions/${payload.prediction_id}/feedback`,
        payload,
        { withCredentials: true }
      );
      setFeedbackPred(null);
      refetch();
    } catch (e) {
      alert("Failed to submit feedback: " + (e.response?.data?.detail || e.message));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-gray-400 hover:text-gray-700 text-sm">← Back</Link>
          <h1 className="text-xl font-bold text-gray-900">Prediction History</h1>
        </div>
        <Link to="/predict" className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium">
          + New Prediction
        </Link>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">

        {/* Filters */}
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6 flex flex-wrap gap-4 items-center">
          <div>
            <label className="text-sm text-gray-600 mr-2">Severity:</label>
            <select
              value={severity}
              onChange={(e) => { setSeverity(e.target.value); setPage(1); }}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
            >
              <option value="">All</option>
              {["healthy", "mild", "moderate", "severe", "critical"].map((s) => (
                <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
              ))}
            </select>
          </div>
          {data && (
            <span className="text-sm text-gray-500 ml-auto">
              {data.total.toLocaleString()} total predictions
            </span>
          )}
        </div>

        {/* Table */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          {isLoading ? (
            <div className="p-16 text-center text-gray-400">
              <div className="text-4xl mb-3 animate-pulse">🌿</div>
              <p>Loading predictions...</p>
            </div>
          ) : isError ? (
            <div className="p-16 text-center text-red-500">
              Failed to load predictions. Are you logged in?
            </div>
          ) : data?.items?.length === 0 ? (
            <div className="p-16 text-center text-gray-400">
              <div className="text-4xl mb-3">🔬</div>
              <p>No predictions yet.</p>
              <Link to="/predict" className="mt-3 inline-block text-sm text-green-600 hover:underline">
                Make your first prediction →
              </Link>
            </div>
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-xs text-gray-500 uppercase tracking-wider">
                  <th className="px-4 py-3">Image</th>
                  <th className="px-4 py-3">Disease</th>
                  <th className="px-4 py-3">Severity</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3">Time</th>
                  <th className="px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((pred) => (
                  <HistoryRow key={pred.prediction_id} pred={pred} onFeedback={setFeedbackPred} />
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {data && data.total_pages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-6">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm disabled:opacity-40 hover:bg-gray-50"
            >
              ← Prev
            </button>
            <span className="text-sm text-gray-600">Page {page} of {data.total_pages}</span>
            <button
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
              disabled={page === data.total_pages}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm disabled:opacity-40 hover:bg-gray-50"
            >
              Next →
            </button>
          </div>
        )}
      </main>

      {feedbackPred && (
        <FeedbackModal
          prediction={feedbackPred}
          onClose={() => setFeedbackPred(null)}
          onSubmit={handleFeedbackSubmit}
        />
      )}
    </div>
  );
}
