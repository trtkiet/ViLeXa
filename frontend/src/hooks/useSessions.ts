/**
 * Hook for managing chat sessions
 */

import { useState, useCallback, useEffect } from 'react';
import type { ChatSession, SessionDetailResponse } from '../types/session';
import { sessionService } from '../services/sessionService';
import { useAuth } from '../contexts/AuthContext';
import { ApiError } from '../services/api';

export interface SessionError {
  message: string;
  operation: 'fetch' | 'create' | 'delete' | 'rename' | 'get';
  sessionId?: string;
}

export function useSessions() {
  const { isAuthenticated } = useAuth();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<SessionError | null>(null);

  // Clear error after timeout
  useEffect(() => {
    if (error) {
      const timeout = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timeout);
    }
  }, [error]);

  /**
   * Fetch all sessions
   */
  const fetchSessions = useCallback(async () => {
    if (!isAuthenticated) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await sessionService.getSessions();
      setSessions(response.sessions);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.isNetworkError
          ? 'Cannot connect to server'
          : err.detail
        : 'Failed to load sessions';
      setError({ message, operation: 'fetch' });
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  // Fetch sessions when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchSessions();
    } else {
      setSessions([]);
      setActiveSessionId(null);
    }
  }, [isAuthenticated, fetchSessions]);

  /**
   * Create a new session
   */
  const createSession = useCallback(async (title?: string): Promise<ChatSession | null> => {
    try {
      const newSession = await sessionService.createSession({ title });
      setSessions(prev => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      return newSession;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.isNetworkError
          ? 'Cannot connect to server'
          : err.detail
        : 'Failed to create session';
      setError({ message, operation: 'create' });
      return null;
    }
  }, []);

  /**
   * Get session with messages
   */
  const getSession = useCallback(async (sessionId: string): Promise<SessionDetailResponse> => {
    try {
      return await sessionService.getSession(sessionId);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.status === 404
          ? 'Session not found'
          : err.isNetworkError
          ? 'Cannot connect to server'
          : err.detail
        : 'Failed to load session';
      setError({ message, operation: 'get', sessionId });
      throw err;
    }
  }, []);

  /**
   * Delete a session
   */
  const deleteSession = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      await sessionService.deleteSession(sessionId);
      setSessions(prev => prev.filter(s => s.id !== sessionId));

      // If deleted active session, clear it
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
      }
      return true;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.status === 404
          ? 'Session not found'
          : err.isNetworkError
          ? 'Cannot connect to server'
          : err.detail
        : 'Failed to delete session';
      setError({ message, operation: 'delete', sessionId });
      return false;
    }
  }, [activeSessionId]);

  /**
   * Rename a session
   */
  const renameSession = useCallback(async (sessionId: string, title: string): Promise<boolean> => {
    if (!title.trim()) {
      setError({ message: 'Title cannot be empty', operation: 'rename', sessionId });
      return false;
    }

    try {
      const updated = await sessionService.updateSession(sessionId, { title: title.trim() });
      setSessions(prev =>
        prev.map(s => (s.id === sessionId ? { ...s, title: updated.title } : s))
      );
      return true;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.status === 404
          ? 'Session not found'
          : err.isNetworkError
          ? 'Cannot connect to server'
          : err.detail
        : 'Failed to rename session';
      setError({ message, operation: 'rename', sessionId });
      return false;
    }
  }, []);

  /**
   * Select a session as active
   */
  const selectSession = useCallback((sessionId: string | null) => {
    setActiveSessionId(sessionId);
  }, []);

  /**
   * Update a session in the list (e.g., after chat message)
   */
  const updateSessionInList = useCallback((sessionId: string, updates: Partial<ChatSession>) => {
    setSessions(prev =>
      prev.map(s => (s.id === sessionId ? { ...s, ...updates } : s))
    );
  }, []);

  /**
   * Clear error manually
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    sessions,
    activeSessionId,
    isLoading,
    error,
    fetchSessions,
    createSession,
    getSession,
    deleteSession,
    renameSession,
    selectSession,
    updateSessionInList,
    clearError,
  };
}
