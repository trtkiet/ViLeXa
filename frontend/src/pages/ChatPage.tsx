import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ChatApiResponse, Message, LawSource } from '../types/index';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const CHAT_ENDPOINT = `${API_BASE_URL}/api/v1/chat`;

// Extended Message type with error state for retry functionality
interface ExtendedMessage extends Message {
  isError?: boolean;
  errorType?: 'network' | 'server' | 'unknown';
}

// Typing indicator component
const TypingIndicator: React.FC = () => (
  <div className="flex gap-4 justify-start">
    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0">
      AI
    </div>
    <div className="flex items-center gap-1 px-4 py-3">
      <div className="flex gap-1">
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
      </div>
      <span className="ml-2 text-sm text-slate-500">Đang suy nghĩ...</span>
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

export const ChatPage: React.FC = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ExtendedMessage[]>([
    { role: 'ai', text: 'Chào bạn, tôi là trợ lý pháp luật ảo. Tôi có thể giúp gì cho bạn hôm nay?' }
  ]);
  const [composer, setComposer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Navigate to lookup page with source information
  const handleSourceClick = (source: LawSource) => {
    const params = new URLSearchParams({
      law_id: source.law_id,
      article: source.article,
    });
    navigate(`/lookup?${params.toString()}`);
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
  const sendMessageToAPI = useCallback(async (messageText: string) => {
    setIsLoading(true);
    setLastFailedMessage(null);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorType = response.status >= 500 ? 'server' : 'unknown';
        const errorMessage = response.status === 500
          ? 'Máy chủ đang gặp sự cố. Vui lòng thử lại sau.'
          : response.status === 503
          ? 'Dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau.'
          : `Có lỗi xảy ra (mã lỗi: ${response.status}). Vui lòng thử lại.`;

        setLastFailedMessage(messageText);
        setMessages(prev => [...prev, {
          role: 'ai',
          text: errorMessage,
          isError: true,
          errorType
        }]);
        return;
      }

      const data = (await response.json()) as ChatApiResponse;
      const reply = data.reply ?? data.answer ?? 'Hiện không có phản hồi từ hệ thống.';
      const sources = data.sources ?? [];

      setMessages(prev => [...prev, { role: 'ai', text: reply, sources: sources }]);
    } catch (error) {
      let errorMessage = 'Đã xảy ra lỗi không xác định. Vui lòng thử lại.';
      let errorType: 'network' | 'server' | 'unknown' = 'unknown';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Yêu cầu đã hết thời gian chờ. Vui lòng thử lại.';
          errorType = 'network';
        } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          errorMessage = 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng.';
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
  }, []);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!composer.trim() || isLoading) return;

    const messageText = composer.trim();
    const userMessage: ExtendedMessage = { role: 'user', text: messageText };
    setMessages(prev => [...prev, userMessage]);
    setComposer('');

    await sendMessageToAPI(messageText);
  };

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

    sendMessageToAPI(lastFailedMessage);
  }, [lastFailedMessage, isLoading, sendMessageToAPI]);

  return (
    <div className="flex h-screen bg-white text-slate-800 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-slate-50 border-r border-slate-200 flex flex-col hidden md:flex">
        <div className="p-4">
          <button className="flex items-center gap-2 px-4 py-3 bg-slate-200 hover:bg-slate-300 rounded-full text-sm font-medium transition-colors w-full text-slate-700">
            <span className="text-lg">+</span> Cuộc trò chuyện mới
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2">
          <div className="px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">Gần đây</div>
          <div className="space-y-1">
            <button className="w-full text-left px-4 py-2 text-sm text-slate-700 hover:bg-slate-200 rounded-lg truncate">
              Luật Lao động 2019
            </button>
            <button className="w-full text-left px-4 py-2 text-sm text-slate-700 hover:bg-slate-200 rounded-lg truncate">
              Quy định về thử việc
            </button>
          </div>
        </div>
        <div className="p-4 border-t border-slate-200 text-xs text-slate-500">
          Legal Assistant v0.1
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Header */}
        <div className="h-14 border-b border-slate-100 flex items-center justify-between px-6">
          <div className="font-medium text-slate-700">Vietnamese Law Assistant</div>
          <div className="text-sm text-slate-500">Gemini 1.5 Pro</div>
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 pb-32">
          <div className="max-w-3xl mx-auto space-y-8">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'ai' && (
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white shrink-0 ${
                    msg.isError ? 'bg-red-500' : 'bg-blue-600'
                  }`}>
                    {msg.isError ? '!' : 'AI'}
                  </div>
                )}

                <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`prose prose-slate max-w-none rounded-2xl px-5 py-3 ${
                    msg.role === 'user'
                      ? 'bg-slate-100 text-slate-800 rounded-br-none'
                      : msg.isError
                      ? 'bg-red-50 text-red-700 border border-red-200'
                      : 'bg-transparent text-slate-800 px-0 py-0'
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
                      Thử lại
                    </button>
                  )}

                  {/* Sources / Citations */}
                  {msg.sources && msg.sources.length > 0 && !msg.isError && (
                    <div className="mt-4 space-y-2 w-full">
                      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Nguồn tham khảo</div>
                      <div className="grid grid-cols-1 gap-2">
                        {msg.sources.map((source, sIdx) => (
                          <div
                            key={sIdx}
                            onClick={() => handleSourceClick(source)}
                            className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-sm hover:bg-slate-100 transition-colors cursor-pointer"
                          >
                            <div className="font-medium text-blue-700 flex items-center gap-2">
                              <span className="text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">
                                {source.article}
                              </span>
                              {source.article_title}
                            </div>
                            <div className="text-slate-600 text-xs mt-1 line-clamp-2">
                              {source.source_text}
                            </div>
                            <div className="mt-1 text-[10px] text-slate-400">
                              {source.law_id} • {source.chapter}
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
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-10 pb-6 px-4">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative flex items-center gap-2 bg-slate-100 rounded-full p-2 pl-4 border border-slate-200 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all shadow-sm">
              <input
                ref={inputRef}
                className="flex-1 bg-transparent border-none outline-none text-slate-700 placeholder-slate-400 disabled:cursor-not-allowed"
                value={composer}
                onChange={(e) => setComposer(e.target.value)}
                placeholder={isLoading ? "Đang chờ phản hồi..." : "Hỏi bất cứ điều gì về luật pháp Việt Nam..."}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !composer.trim()}
                className={`p-2 rounded-full transition-colors ${
                  isLoading
                    ? 'bg-blue-400 text-white cursor-wait'
                    : composer.trim()
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-slate-300 text-slate-500 cursor-not-allowed'
                }`}
                title={isLoading ? 'Đang xử lý...' : composer.trim() ? 'Gửi tin nhắn' : 'Nhập tin nhắn để gửi'}
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
            <div className="text-center text-xs text-slate-400 mt-2">
              {isLoading ? (
                <span className="text-blue-500">AI đang xử lý yêu cầu của bạn...</span>
              ) : (
                'AI có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.'
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};