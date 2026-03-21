/**
 * Frontend API Service Layer
 * ───────────────────────────
 * Centralized axios instance with:
 *  - Automatic JWT injection
 *  - Token refresh on 401
 *  - Request/response interceptors
 *  - Typed API functions for all endpoints
 */

import axios from "axios";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
const API_V1 = `${BASE_URL}/api/v1`;

// ── Axios instance ────────────────────────────────────────────────────────────
export const apiClient = axios.create({
  baseURL: API_V1,
  timeout: 120_000,     // 2 min — prediction can be slow on CPU
  withCredentials: true,
});

// ── Request interceptor: inject access token ──────────────────────────────────
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response interceptor: auto-refresh token on 401 ──────────────────────────
let isRefreshing = false;
let pendingRequests = [];

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // Queue request until refresh completes
        return new Promise((resolve, reject) => {
          pendingRequests.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return apiClient(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        if (!refreshToken) throw new Error("No refresh token");

        const { data } = await axios.post(`${API_V1}/auth/refresh`, null, {
          headers: { Authorization: `Bearer ${refreshToken}` },
        });

        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);

        // Resume queued requests
        pendingRequests.forEach(({ resolve }) => resolve(data.access_token));
        pendingRequests = [];

        originalRequest.headers.Authorization = `Bearer ${data.access_token}`;
        return apiClient(originalRequest);
      } catch (refreshError) {
        // Refresh failed — clear tokens and redirect to login
        pendingRequests.forEach(({ reject }) => reject(refreshError));
        pendingRequests = [];
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);


// ── Auth ──────────────────────────────────────────────────────────────────────

export const authAPI = {
  login: async (email, password) => {
    const { data } = await apiClient.post("/auth/login", { email, password });
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    return data;
  },

  register: async (payload) => {
    const { data } = await apiClient.post("/auth/register", payload);
    return data;
  },

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    window.location.href = "/login";
  },

  getMe: async () => {
    const { data } = await apiClient.get("/auth/me");
    return data;
  },
};


// ── Predictions ───────────────────────────────────────────────────────────────

export const predictionsAPI = {
  predict: async (imageFile, { cropHint, topK = 5, asyncMode = false,recommendationMode = "db" } = {}) => {
    const form = new FormData();
    form.append("image", imageFile);
    if (cropHint) form.append("crop_hint", cropHint);
    form.append("top_k", String(topK));
    form.append("async_mode", String(asyncMode));
    form.append("recommendation_mode", recommendationMode);

    const { data } = await apiClient.post("/predictions/", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },

  getPrediction: async (predictionId) => {
    const { data } = await apiClient.get(`/predictions/${predictionId}`);
    return data;
  },

  listPredictions: async ({ page = 1, pageSize = 20, severity, diseaseCode } = {}) => {
    const params = new URLSearchParams({ page, page_size: pageSize });
    if (severity) params.append("severity", severity);
    if (diseaseCode) params.append("disease_code", diseaseCode);
    const { data } = await apiClient.get(`/predictions/?${params}`);
    return data;
  },

  submitFeedback: async (predictionId, feedback) => {
    const { data } = await apiClient.post(`/predictions/${predictionId}/feedback`, feedback);
    return data;
  },

  // Poll async prediction until completed or failed
  pollUntilDone: async (predictionId, { intervalMs = 2000, maxAttempts = 60 } = {}) => {
    for (let i = 0; i < maxAttempts; i++) {
      const pred = await predictionsAPI.getPrediction(predictionId);
      if (pred.status === "completed") return pred;
      if (pred.status === "failed") throw new Error(pred.error_message || "Prediction failed");
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error("Prediction timed out");
  },
};


// ── Diseases ──────────────────────────────────────────────────────────────────

export const diseasesAPI = {
  list: async ({ cropName, pathogenType, skip = 0, limit = 100 } = {}) => {
    const params = new URLSearchParams({ skip, limit });
    if (cropName) params.append("crop_name", cropName);
    if (pathogenType) params.append("pathogen_type", pathogenType);
    const { data } = await apiClient.get(`/diseases/?${params}`);
    return data;
  },

  get: async (diseaseCode) => {
    const { data } = await apiClient.get(`/diseases/${diseaseCode}`);
    return data;
  },

  getCrops: async () => {
    const { data } = await apiClient.get("/diseases/crops");
    return data.crops;
  },

  search: async (query) => {
    const { data } = await apiClient.get(`/diseases/search?q=${encodeURIComponent(query)}`);
    return data;
  },
};


// ── Users ─────────────────────────────────────────────────────────────────────

export const usersAPI = {
  getProfile: async () => {
    const { data } = await apiClient.get("/users/me");
    return data;
  },

  updateProfile: async (payload) => {
    const { data } = await apiClient.patch("/users/me", payload);
    return data;
  },

  getStats: async () => {
    const { data } = await apiClient.get("/users/me/stats");
    return data;
  },

  createAPIKey: async (payload) => {
    const { data } = await apiClient.post("/users/me/api-keys", payload);
    return data;
  },

  listAPIKeys: async () => {
    const { data } = await apiClient.get("/users/me/api-keys");
    return data;
  },

  revokeAPIKey: async (keyId) => {
    await apiClient.delete(`/users/me/api-keys/${keyId}`);
  },
};


// ── Admin ─────────────────────────────────────────────────────────────────────

export const adminAPI = {
  getStats: async () => {
    const { data } = await apiClient.get("/admin/stats");
    return data;
  },

  listModels: async () => {
    const { data } = await apiClient.get("/admin/models");
    return data;
  },

  activateModel: async (modelId) => {
    const { data } = await apiClient.post(`/admin/models/${modelId}/activate`);
    return data;
  },

  triggerRetrain: async () => {
    const { data } = await apiClient.post("/admin/retrain");
    return data;
  },
};


// ── Helpers ───────────────────────────────────────────────────────────────────

export const isAuthenticated = () => !!localStorage.getItem("access_token");

export const formatConfidence = (conf) =>
  `${(conf * 100).toFixed(1)}%`;

export const formatDiseaseName = (code) => {
  const parts = code.split("___");
  if (parts.length === 2) {
    const crop    = parts[0].replace(/_/g, " ");
    const disease = parts[1].replace(/_/g, " ");
    return `${crop.charAt(0).toUpperCase() + crop.slice(1)} — ${disease.charAt(0).toUpperCase() + disease.slice(1)}`;
  }
  return code.replace(/_/g, " ");
};
