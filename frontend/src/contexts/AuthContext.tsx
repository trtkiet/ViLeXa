/**
 * Authentication context for global auth state management
 */

import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react';
import type { User, AuthState, LoginPayload, RegisterPayload } from '../types/auth';
import { authService } from '../services/authService';

interface AuthContextType extends AuthState {
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = user !== null;

  /**
   * Check authentication status on mount
   */
  const checkAuth = useCallback(async () => {
    if (!authService.isLoggedIn()) {
      setIsLoading(false);
      return;
    }

    try {
      const currentUser = await authService.getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      // Token invalid or expired
      authService.logout();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  /**
   * Listen for token expiry events from API calls
   */
  useEffect(() => {
    const handleTokenExpired = () => {
      authService.logout();
      setUser(null);
    };

    window.addEventListener('auth:token-expired', handleTokenExpired);
    return () => {
      window.removeEventListener('auth:token-expired', handleTokenExpired);
    };
  }, []);

  /**
   * Login and fetch user info
   */
  const login = useCallback(async (payload: LoginPayload) => {
    setIsLoading(true);
    try {
      await authService.login(payload);
      const currentUser = await authService.getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      authService.logout();
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Register a new user (does not auto-login)
   */
  const register = useCallback(async (payload: RegisterPayload) => {
    await authService.register(payload);
    // Note: Does not auto-login, user needs to login manually
  }, []);

  /**
   * Logout and clear state
   */
  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
  }, []);

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Hook to use auth context
 */
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
