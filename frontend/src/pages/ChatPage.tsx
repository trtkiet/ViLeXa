import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ChatApiResponse, Message, LawSource } from '../types/index';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const CHAT_ENDPOINT = `${API_BASE_URL}/api/v1/chat`;

export const ChatPage: React.FC = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', text: 'Chào bạn, tôi là trợ lý pháp luật ảo. Tôi có thể giúp gì cho bạn hôm nay?' }
  ]);
  const [composer, setComposer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const scrollRef = useRef<HTMLDivElement>(null);

  // Navigate to lookup page with source information
  const handleSourceClick = (source: LawSource) => {
    const params = new URLSearchParams({
      law_id: source.law_id,
      article: source.article,
    });
    navigate(`/lookup?${params.toString()}`);
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!composer.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', text: composer };
    setMessages(prev => [...prev, userMessage]);
    setComposer('');
    setIsLoading(true);

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.text }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = (await response.json()) as ChatApiResponse;
      const reply = data.reply ?? data.answer ?? 'Hiện không có phản hồi.';
      const sources = data.sources ?? [];

      setMessages(prev => [...prev, { role: 'ai', text: reply, sources: sources }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: 'Lỗi kết nối máy chủ.' }]);
    } finally {
      setIsLoading(false);
    }
  };

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
                  <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0">
                    AI
                  </div>
                )}
                
                <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`prose prose-slate max-w-none rounded-2xl px-5 py-3 ${
                    msg.role === 'user' 
                      ? 'bg-slate-100 text-slate-800 rounded-br-none' 
                      : 'bg-transparent text-slate-800 px-0 py-0'
                  }`}>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>

                  {/* Sources / Citations */}
                  {msg.sources && msg.sources.length > 0 && (
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
          </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-10 pb-6 px-4">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative flex items-center gap-2 bg-slate-100 rounded-full p-2 pl-4 border border-slate-200 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all shadow-sm">
              <input
                className="flex-1 bg-transparent border-none outline-none text-slate-700 placeholder-slate-400"
                value={composer}
                onChange={(e) => setComposer(e.target.value)}
                placeholder="Hỏi bất cứ điều gì về luật pháp Việt Nam..."
                disabled={isLoading}
              />
              <button 
                type="submit" 
                disabled={isLoading || !composer.trim()}
                className={`p-2 rounded-full transition-colors ${
                  composer.trim() ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-300 text-slate-500 cursor-not-allowed'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                  <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                </svg>
              </button>
            </form>
            <div className="text-center text-xs text-slate-400 mt-2">
              AI có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};