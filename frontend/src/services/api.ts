/**
 * Base API service with authentication headers
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const TOKEN_KEY = 'vla_access_token';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export function getAuthHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class ApiError extends Error {
  status: number;
  detail: string;
  isNetworkError: boolean;
  isAuthError: boolean;

  constructor(status: number, detail: string, isNetworkError = false) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.isNetworkError = isNetworkError;
    this.isAuthError = status === 401 || status === 403;
  }
}

// Extract user-friendly error message from response
async function extractErrorMessage(response: Response): Promise<string> {
  const defaultMessages: Record<number, string> = {
    400: 'Invalid request. Please check your input.',
    401: 'Session expired. Please login again.',
    403: 'You do not have permission to perform this action.',
    404: 'The requested resource was not found.',
    422: 'Invalid data provided. Please check your input.',
    429: 'Too many requests. Please wait a moment.',
    500: 'Server error. Please try again later.',
    502: 'Service temporarily unavailable. Please try again.',
    503: 'Service temporarily unavailable. Please try again.',
    504: 'Request timed out. Please try again.',
  };

  try {
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const errorData = await response.json();
      // Handle various error response formats
      if (typeof errorData.detail === 'string') {
        return errorData.detail;
      }
      if (Array.isArray(errorData.detail)) {
        // Pydantic validation errors
        return errorData.detail
          .map((err: { loc?: string[]; msg?: string }) => {
            const field = err.loc?.slice(-1)[0] || 'field';
            return `${field}: ${err.msg || 'invalid'}`;
          })
          .join('; ');
      }
      if (errorData.message) {
        return errorData.message;
      }
      if (errorData.error) {
        return errorData.error;
      }
    }
  } catch {
    // Ignore JSON parse errors
  }

  return defaultMessages[response.status] || `Request failed (error ${response.status})`;
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeader(),
        ...options.headers,
      },
    });
  } catch (error) {
    // Network errors (no connection, CORS, etc.)
    const message = error instanceof Error
      ? error.message.includes('Failed to fetch') || error.message.includes('NetworkError')
        ? 'Cannot connect to server. Please check your connection.'
        : `Network error: ${error.message}`
      : 'Network error. Please check your connection.';
    throw new ApiError(0, message, true);
  }

  if (!response.ok) {
    const detail = await extractErrorMessage(response);
    const apiError = new ApiError(response.status, detail);

    // Handle token expiry - trigger logout event
    if (apiError.isAuthError && response.status === 401) {
      // Dispatch custom event for auth context to handle
      window.dispatchEvent(new CustomEvent('auth:token-expired'));
    }

    throw apiError;
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return await response.json();
  } catch {
    // Response was OK but not valid JSON
    return undefined as T;
  }
}

export { API_BASE_URL };
