import React, { useState, useRef, useEffect } from 'react';
import { Panel, Badge, Input, PrimaryButton, Button } from '../components/ui';
import type { ChatApiResponse, Message, LawSource } from '../types/index';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const CHAT_ENDPOINT = `${API_BASE_URL}/api/v1/chat`;

export const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', text: 'Chào bạn, tôi có thể giúp trả lời câu hỏi pháp lý và trích dẫn điều khoản liên quan.' }
  ]);
  const [composer, setComposer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState<LawSource[]>([]);
  
  const scrollRef = useRef<HTMLDivElement>(null);

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

      setMessages(prev => [...prev, { role: 'ai', text: reply }]);
      setSources(data.sources ?? []);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: 'Lỗi kết nối máy chủ.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-5 gap-6">
      <Panel className="col-span-3 p-6 flex flex-col h-[600px]">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Assistant</h2>
          <Badge className={isLoading ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}>
            {isLoading ? 'Typing...' : 'Online'}
          </Badge>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto flex flex-col gap-4 mb-4 pr-2">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${
                msg.role === 'user' ? 'bg-slate-900 text-white' : 'bg-slate-100'
              }`}>
                {msg.text}
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={handleSendMessage} className="flex gap-2 border-t pt-4">
          <Input 
            value={composer} 
            onChange={(e) => setComposer(e.target.value)} 
            placeholder="Hỏi về luật lao động..."
            disabled={isLoading}
          />
          <PrimaryButton type="submit" disabled={isLoading}>Gửi</PrimaryButton>
        </form>
      </Panel>

      <div className="col-span-2 space-y-4">
        <Panel className="p-5">
          <p className="text-xs font-bold uppercase text-slate-500 mb-3">Sources Cited</p>
          <div className="space-y-3">
            {sources.length > 0 ? sources.map((s, i) => (
              <div key={i} className="text-sm p-3 border rounded-lg bg-slate-50">
                <div className="font-bold">{s.title} ({s.ref})</div>
                <div className="text-slate-600 italic mt-1 text-xs">"{s.snippet}"</div>
              </div>
            )) : <p className="text-sm text-slate-400">No citations yet.</p>}
          </div>
        </Panel>
      </div>
    </div>
  );
};