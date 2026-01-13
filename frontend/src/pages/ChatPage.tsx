import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, NavLink } from 'react-router-dom';
import type { LawSource } from '../types/index';
import type { SessionMessage } from '../types/session';
import ReactMarkdown from 'react-markdown';
import { Sidebar } from '../components/layout/Sidebar';
import { useSessions, type SessionError } from '../hooks/useSessions';
import { getAuthHeader, API_BASE_URL } from '../services/api';

// Toast notification component
const Toast: React.FC<{ error: SessionError; onDismiss: () => void }> = ({ error, onDismiss }) => (
  <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 z-50 animate-in fade-in slide-in-from-bottom-4 duration-200">
    <div className="bg-red-600 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 max-w-md">
      <svg className="w-5 h-5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
      <span className="text-sm">{error.message}</span>
      <button onClick={onDismiss} className="ml-2 p-1 hover:bg-red-700 rounded transition-colors">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  </div>
);

// Extended Message type with error state for retry functionality
interface ExtendedMessage {
  id?: number;
  role: 'user' | 'ai';
  text: string;
  sources?: LawSource[];
  isError?: boolean;
  errorType?: 'network' | 'server' | 'unknown';
}

// Chat API response
interface ChatApiResponse {
  reply: string;
  sources?: LawSource[];
  session_id: string;
}

// Typing indicator component
const TypingIndicator: React.FC = () => (
  <div className="flex gap-4 justify-start">
    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white text-xs font-medium shrink-0">
      VA
    </div>
    <div className="flex items-center gap-1 px-4 py-3">
      <div className="flex gap-1">
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
      </div>
      <span className="ml-2 text-sm text-gray-500">Thinking...</span>
    </div>
  </div>
);

// Loading spinner for button
const LoadingSpinner: React.FC = () => (
  <svg className="w-5 h-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

// Convert session messages to extended messages
function convertSessionMessages(messages: SessionMessage[]): ExtendedMessage[] {
  return messages.map(msg => ({
    id: msg.id,
    role: msg.role,
    text: msg.content,
    sources: msg.sources as LawSource[],
  }));
}

export const ChatPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    sessions,
    activeSessionId,
    isLoading: isSessionsLoading,
    error: sessionError,
    fetchSessions,
    getSession,
    deleteSession,
    renameSession,
    selectSession,
    updateSessionInList,
    clearError: clearSessionError,
  } = useSessions();

  const [messages, setMessages] = useState<ExtendedMessage[]>([]);
  const [composer, setComposer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load messages when active session changes
  useEffect(() => {
    const loadMessages = async () => {
      if (!activeSessionId) {
        setMessages([]);
        return;
      }

      setIsLoadingMessages(true);
      try {
        const sessionDetail = await getSession(activeSessionId);
        setMessages(convertSessionMessages(sessionDetail.messages));
      } catch (error) {
        console.error('Failed to load messages:', error);
        setMessages([]);
      } finally {
        setIsLoadingMessages(false);
      }
    };

    loadMessages();
  }, [activeSessionId, getSession]);

  // Navigate to lookup page by searching source content
  const handleSourceClick = async (source: LawSource) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/documents/search-by-content`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader(),
        },
        body: JSON.stringify({ source_text: source.source_text }),
      });

      if (response.ok) {
        const document = await response.json();
        const params = new URLSearchParams({
          id: document.id,
          article: source.article,
        });
        navigate(`/lookup?${params.toString()}`);
      } else {
        console.error('Document not found');
      }
    } catch (error) {
      console.error('Error searching document:', error);
    }
  };

  // Auto-scroll to bottom when messages change or loading state changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Auto-focus input after sending message
  useEffect(() => {
    if (!isLoading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isLoading]);

  // Send message to API
  const sendMessageToAPI = useCallback(async (messageText: string, sessionId: string | null) => {
    setIsLoading(true);
    setLastFailedMessage(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader(),
        },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorType = response.status >= 500 ? 'server' : 'unknown';
        const errorMessage = response.status === 500
          ? 'Server error. Please try again later.'
          : response.status === 503
          ? 'Service temporarily unavailable. Please try again later.'
          : response.status === 401
          ? 'Session expired. Please login again.'
          : `An error occurred (code: ${response.status}). Please try again.`;

        setLastFailedMessage(messageText);
        setMessages(prev => [...prev, {
          role: 'ai',
          text: errorMessage,
          isError: true,
          errorType
        }]);
        return;
      }

      const data = await response.json() as ChatApiResponse;

      // If this was a new session, select it and refresh sessions list
      if (!sessionId && data.session_id) {
        selectSession(data.session_id);
        fetchSessions();
      }

      // Update session in list to move it to top
      if (data.session_id) {
        updateSessionInList(data.session_id, {
          updated_at: new Date().toISOString(),
        });
      }

      setMessages(prev => [...prev, {
        role: 'ai',
        text: data.reply,
        sources: data.sources,
      }]);
    } catch (error) {
      let errorMessage = 'An unknown error occurred. Please try again.';
      let errorType: 'network' | 'server' | 'unknown' = 'unknown';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timed out. Please try again.';
          errorType = 'network';
        } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          errorMessage = 'Cannot connect to server. Please check your connection.';
          errorType = 'network';
        }
      }

      setLastFailedMessage(messageText);
      setMessages(prev => [...prev, {
        role: 'ai',
        text: errorMessage,
        isError: true,
        errorType
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [selectSession, fetchSessions, updateSessionInList]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!composer.trim() || isLoading) return;

    const messageText = composer.trim();
    const userMessage: ExtendedMessage = { role: 'user', text: messageText };
    setMessages(prev => [...prev, userMessage]);
    setComposer('');

    await sendMessageToAPI(messageText, activeSessionId);
  };

  // Start new conversation
  const handleNewConversation = useCallback(() => {
    selectSession(null);
    setMessages([]);
    setSidebarOpen(false);
  }, [selectSession]);

  // Select a session
  const handleSelectSession = useCallback((sessionId: string | null) => {
    selectSession(sessionId);
    setSidebarOpen(false);
  }, [selectSession]);

  // Retry failed message
  const handleRetry = useCallback(() => {
    if (!lastFailedMessage || isLoading) return;

    // Remove the last error message
    setMessages(prev => {
      const newMessages = [...prev];
      if (newMessages.length > 0 && newMessages[newMessages.length - 1].isError) {
        newMessages.pop();
      }
      return newMessages;
    });

    sendMessageToAPI(lastFailedMessage, activeSessionId);
  }, [lastFailedMessage, isLoading, sendMessageToAPI, activeSessionId]);

  return (
    <div className="flex h-screen bg-white text-gray-800 font-sans">
      {/* Session error toast */}
      {sessionError && (
        <Toast error={sessionError} onDismiss={clearSessionError} />
      )}

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 transform transition-transform duration-200
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:relative md:translate-x-0
      `}>
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewConversation}
          onDeleteSession={deleteSession}
          onRenameSession={renameSession}
          isLoading={isSessionsLoading}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative min-w-0">
        {/* Header */}
        <div className="h-14 border-b border-gray-100 flex items-center justify-between px-4 md:px-6 shrink-0">
          <div className="flex items-center gap-3">
            {/* Mobile menu button */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-1.5 -ml-1.5 hover:bg-gray-100 rounded-lg"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <span className="bg-gray-900 text-white px-2 py-0.5 rounded text-sm font-bold">VA</span>
              <span className="font-medium text-gray-900 hidden sm:inline">Law Assistant</span>
            </div>
          </div>
          <nav className="flex items-center gap-1">
            <NavLink
              to="/"
              className={({isActive}) =>
                `px-3 py-1.5 text-sm rounded-lg transition-colors ${
                  isActive ? 'bg-gray-100 font-medium' : 'hover:bg-gray-50'
                }`
              }
            >
              Chat
            </NavLink>
            <NavLink
              to="/lookup"
              className={({isActive}) =>
                `px-3 py-1.5 text-sm rounded-lg transition-colors ${
                  isActive ? 'bg-gray-100 font-medium' : 'hover:bg-gray-50'
                }`
              }
            >
              Lookup
            </NavLink>
          </nav>
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 pb-36">
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Welcome message when no session selected and no messages */}
            {!activeSessionId && messages.length === 0 && !isLoadingMessages && (
              <div className="text-center py-12">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white text-xl font-bold mx-auto mb-4">
                  VA
                </div>
                <h2 className="text-xl font-medium text-gray-900 mb-2">
                  Vietnamese Law Assistant
                </h2>
                <p className="text-gray-500 max-w-md mx-auto">
                  Ask me anything about Vietnamese law. I'll help you find relevant legal information and explain complex regulations.
                </p>
              </div>
            )}

            {/* Loading messages indicator */}
            {isLoadingMessages && (
              <div className="flex justify-center py-8">
                <div className="flex items-center gap-2 text-gray-500">
                  <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                  <span className="text-sm">Loading messages...</span>
                </div>
              </div>
            )}

            {/* Messages */}
            {!isLoadingMessages && messages.map((msg, idx) => (
              <div
                key={msg.id || idx}
                className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-200`}
              >
                {msg.role === 'ai' && (
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-medium shrink-0 ${
                    msg.isError
                      ? 'bg-red-500'
                      : 'bg-gradient-to-br from-blue-500 to-blue-600'
                  }`}>
                    {msg.isError ? '!' : 'VA'}
                  </div>
                )}

                <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`prose prose-gray prose-sm max-w-none rounded-2xl px-4 py-2.5 ${
                    msg.role === 'user'
                      ? 'bg-gray-100 text-gray-800 rounded-br-sm'
                      : msg.isError
                      ? 'bg-red-50 text-red-700 border border-red-200'
                      : 'bg-transparent text-gray-800 px-0 py-0'
                  }`}>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>

                  {/* Retry button for error messages */}
                  {msg.isError && idx === messages.length - 1 && lastFailedMessage && (
                    <button
                      onClick={handleRetry}
                      disabled={isLoading}
                      className="mt-2 flex items-center gap-2 px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clipRule="evenodd" />
                      </svg>
                      Retry
                    </button>
                  )}

                  {/* Sources / Citations */}
                  {msg.sources && msg.sources.length > 0 && !msg.isError && (
                    <div className="mt-4 space-y-2 w-full">
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Sources</div>
                      <div className="grid grid-cols-1 gap-2">
                        {msg.sources.map((source, sIdx) => (
                          <div
                            key={sIdx}
                            onClick={() => handleSourceClick(source)}
                            className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm hover:bg-gray-100 hover:border-gray-300 transition-colors cursor-pointer"
                          >
                            <div className="font-medium text-blue-700 flex items-center gap-2">
                              <span className="text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded font-medium">
                                {source.article}
                              </span>
                              <span className="truncate">{source.article_title}</span>
                            </div>
                            <div className="text-gray-600 text-xs mt-1.5 line-clamp-2">
                              {source.source_text}
                            </div>
                            <div className="mt-1.5 text-[10px] text-gray-400">
                              {source.chapter}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Typing indicator when loading */}
            {isLoading && <TypingIndicator />}
          </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-8 pb-4 px-4">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative flex items-center gap-2 bg-gray-50 rounded-2xl p-2 pl-4 border border-gray-200 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all shadow-sm">
              <input
                ref={inputRef}
                className="flex-1 bg-transparent border-none outline-none text-gray-700 placeholder-gray-400 disabled:cursor-not-allowed text-[15px]"
                value={composer}
                onChange={(e) => setComposer(e.target.value)}
                placeholder={isLoading ? "Waiting for response..." : "Ask anything about Vietnamese law..."}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !composer.trim()}
                className={`p-2.5 rounded-xl transition-all ${
                  isLoading
                    ? 'bg-blue-400 text-white cursor-wait'
                    : composer.trim()
                    ? 'bg-gray-900 text-white hover:bg-gray-800'
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                }`}
                title={isLoading ? 'Processing...' : composer.trim() ? 'Send message' : 'Type a message to send'}
              >
                {isLoading ? (
                  <LoadingSpinner />
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                  </svg>
                )}
              </button>
            </form>
            <div className="text-center text-xs text-gray-400 mt-2">
              {isLoading ? (
                <span className="text-blue-500">AI is processing your request...</span>
              ) : (
                'AI may make mistakes. Please verify important information.'
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
