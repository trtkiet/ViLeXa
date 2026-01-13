/**
 * Chat session API service
 */

import { apiRequest } from './api';
import type {
  ChatSession,
  SessionListResponse,
  SessionDetailResponse,
  CreateSessionPayload,
  UpdateSessionPayload,
} from '../types/session';

export const sessionService = {
  /**
   * List all sessions for current user
   */
  async getSessions(): Promise<SessionListResponse> {
    return apiRequest<SessionListResponse>('/api/v1/sessions');
  },

  /**
   * Create a new chat session
   */
  async createSession(payload?: CreateSessionPayload): Promise<ChatSession> {
    return apiRequest<ChatSession>('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    });
  },

  /**
   * Get session with all messages
   */
  async getSession(sessionId: string): Promise<SessionDetailResponse> {
    return apiRequest<SessionDetailResponse>(`/api/v1/sessions/${sessionId}`);
  },

  /**
   * Update session title
   */
  async updateSession(sessionId: string, payload: UpdateSessionPayload): Promise<ChatSession> {
    return apiRequest<ChatSession>(`/api/v1/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<void> {
    await apiRequest<void>(`/api/v1/sessions/${sessionId}`, {
      method: 'DELETE',
    });
  },
};
