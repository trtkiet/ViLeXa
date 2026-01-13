/**
 * Authentication API service
 */

import { apiRequest, setToken, removeToken, getToken } from './api';
import type { User, LoginPayload, RegisterPayload, TokenResponse } from '../types/auth';

export const authService = {
  /**
   * Register a new user
   */
  async register(payload: RegisterPayload): Promise<User> {
    return apiRequest<User>('/api/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Login and store token
   */
  async login(payload: LoginPayload): Promise<TokenResponse> {
    const response = await apiRequest<TokenResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    // Store token in localStorage
    setToken(response.access_token);

    return response;
  },

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<User> {
    return apiRequest<User>('/api/v1/auth/me');
  },

  /**
   * Logout - clear token
   */
  logout(): void {
    removeToken();
  },

  /**
   * Check if user is logged in (has token)
   */
  isLoggedIn(): boolean {
    return getToken() !== null;
  },
};
