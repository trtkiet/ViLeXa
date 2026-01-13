/**
 * Session-related type definitions
 */

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionListResponse {
  sessions: ChatSession[];
  total: number;
}

export interface MessageSource {
  chapter?: string;
  section?: string;
  article?: string;
  article_title?: string;
  clause?: string;
  source_text?: string;
}

export interface SessionMessage {
  id: number;
  role: 'user' | 'ai';
  content: string;
  timestamp: string;
  sources: MessageSource[];
}

export interface SessionDetailResponse {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
}

export interface CreateSessionPayload {
  title?: string;
}

export interface UpdateSessionPayload {
  title: string;
}
