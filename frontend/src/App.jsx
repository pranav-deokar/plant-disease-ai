/**
 * App.jsx — React Application Root
 * React Router v6, React Query, global auth state
 */

import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

// Pages
import PredictPage  from "./pages/PredictPage";
import HistoryPage  from "./pages/HistoryPage";
import DiseasesPage from "./pages/DiseasesPage";
import AdminPage    from "./pages/AdminPage";
import LoginPage    from "./pages/LoginPage";

// Global query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/"           element={<PredictPage />} />
          <Route path="/predict"    element={<PredictPage />} />
          <Route path="/history"    element={<HistoryPage />} />
          <Route path="/diseases"   element={<DiseasesPage />} />
          <Route path="/admin"      element={<AdminPage />} />
          <Route path="/login"      element={<LoginPage />} />
          <Route path="*"           element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
      {process.env.NODE_ENV === "development" && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  );
}
