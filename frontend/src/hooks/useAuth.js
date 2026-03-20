/**
 * useAuth — React hook for authentication state management
 * Wraps authAPI with React Query for caching and automatic refetch.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { authAPI, isAuthenticated } from "../services/api";

export function useAuth() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: user, isLoading, isError } = useQuery({
    queryKey: ["me"],
    queryFn: authAPI.getMe,
    enabled: isAuthenticated(),
    retry: false,
    staleTime: 5 * 60 * 1000,  // 5 minutes
  });

  const loginMutation = useMutation({
    mutationFn: ({ email, password }) => authAPI.login(email, password),
    onSuccess: () => {
      queryClient.invalidateQueries(["me"]);
      navigate("/");
    },
  });

  const logout = () => {
    queryClient.clear();
    authAPI.logout();
  };

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    isAdmin: user?.role === "admin",
    login: loginMutation.mutate,
    loginError: loginMutation.error?.response?.data?.detail,
    isLoggingIn: loginMutation.isPending,
    logout,
  };
}
