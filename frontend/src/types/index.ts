export interface Message {
  role: 'user' | 'ai';
  text: string;
}

export interface LawSource {
  title: string;
  ref: string;
  snippet: string;
}

export interface LawDocument {
  id: string;
  title: string;
  type: string;
  ref: string;
  date: string;
  snippet: string;
  content?: string; // Full text for the preview
}

// API response for the chat endpoint
export interface ChatApiResponse {
  reply?: string;
  answer?: string; // backward compatibility if backend changes
  sources?: LawSource[];
}