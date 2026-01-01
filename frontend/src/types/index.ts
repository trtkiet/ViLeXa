export interface Message {
  role: 'user' | 'ai';
  text: string;
  sources?: LawSource[];
}

export interface LawSource {
  law_id: string;
  chapter: string;
  section: string;
  article: string;
  article_title: string;
  clause: string;
  source_text: string;
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

// Response model for paginated document list
export interface DocumentListResponse {
  items: LawDocument[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Response model for a single article within a law
export interface ArticleResponse {
  article_number: string;
  article_title: string;
  content: string;
  chapter?: string;
  section?: string;
}

// Response model for available document types
export interface TypesResponse {
  types: string[];
}

// API response for the chat endpoint
export interface ChatApiResponse {
  reply?: string;
  answer?: string; // backward compatibility if backend changes
  sources?: LawSource[];
}