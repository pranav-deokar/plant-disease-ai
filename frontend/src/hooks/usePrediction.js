/**
 * usePrediction — React hook for the full prediction flow
 * Handles image upload, polling for async results, and state management.
 */

import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { predictionsAPI } from "../services/api";

export function usePrediction() {
  const queryClient = useQueryClient();
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);

  const mutation = useMutation({
    mutationFn: ({ file, cropHint, topK }) =>
      predictionsAPI.predict(file, { cropHint, topK }),
    onSuccess: () => {
      // Invalidate history so it refreshes
      queryClient.invalidateQueries(["predictions"]);
    },
  });

  const selectImage = useCallback((file) => {
    if (!file) return;
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    mutation.reset();
    return () => URL.revokeObjectURL(url);
  }, []);

  const predict = useCallback(
    ({ cropHint, topK = 5 } = {}) => {
      if (!imageFile) return;
      mutation.mutate({ file: imageFile, cropHint, topK });
    },
    [imageFile, mutation]
  );

  const reset = useCallback(() => {
    setImageFile(null);
    setImagePreview(null);
    mutation.reset();
  }, [mutation]);

  return {
    imageFile,
    imagePreview,
    selectImage,
    predict,
    reset,
    result: mutation.data,
    isPending: mutation.isPending,
    isSuccess: mutation.isSuccess,
    isError: mutation.isError,
    error: mutation.error?.response?.data?.detail || mutation.error?.message,
  };
}
