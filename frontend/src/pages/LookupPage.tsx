import React, { useState, useEffect } from 'react';
import { Panel, Badge, Input, Button } from '../components/ui';
import type { LawDocument, Message } from '../types';

// Mock data for initial view
const MOCK_RESULTS: LawDocument[] = [
  {
    id: 'boll-25',
    title: 'Bộ luật Lao động 2019',
    type: 'Luật',
    ref: 'Điều 25',
    date: '2019',
    snippet: 'Thời gian thử việc tùy thuộc vào tính chất công việc và trình độ chuyên môn của người lao động...',
    content: 'Điều 25. Thời gian thử việc: Thời gian thử việc do hai bên thỏa thuận căn cứ vào tính chất và mức độ phức tạp của công việc nhưng chỉ được thử việc một lần đối với một công việc và bảo đảm điều kiện sau đây: 1. Không quá 180 ngày đối với công việc của người quản lý doanh nghiệp... 2. Không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên...'
  },
  {
    id: 'nd-145-7',
    title: 'Nghị định 145/2020/NĐ-CP',
    type: 'Nghị định',
    ref: 'Điều 7',
    date: '2020',
    snippet: 'Hướng dẫn chi tiết về thử việc, hợp đồng lao động và trách nhiệm thông báo kết quả thử việc...',
    content: 'Điều 7. Thử việc: 1. Khi kết thúc thời gian thử việc, người sử dụng lao động phải thông báo kết quả thử việc cho người lao động. 2. Trường hợp thử việc đạt yêu cầu thì người sử dụng lao động tiếp tục thực hiện hợp đồng lao động đã giao kết...'
  }
];

export const LookupPage: React.FC = () => {
  const [query, setQuery] = useState('thử việc');
  const [results, setResults] = useState<LawDocument[]>(MOCK_RESULTS);
  const [selectedId, setSelectedId] = useState<string | null>(MOCK_RESULTS[0].id);
  const [isLoading, setIsLoading] = useState(false);

  // Function to handle search from backend
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      // Replace with your actual API call
      // const response = await fetch(`YOUR_BACKEND_URL/search?q=${encodeURIComponent(query)}`);
      // const data = await response.json();
      // setResults(data);
      
      // Simulating network delay for demo
      await new Promise(resolve => setTimeout(resolve, 800));
      console.log("Searching for:", query);
      
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setIsLoading(false);
    }
  };

  const activeDoc = results.find((item) => item.id === selectedId);

  return (
    <div className="grid grid-cols-5 gap-6">
      {/* Left Sidebar: Search & List */}
      <div className="col-span-2 flex flex-col gap-4">
        <Panel className="p-5">
          <form onSubmit={handleSearch} className="flex items-center gap-2">
            <Input 
              value={query} 
              onChange={(e) => setQuery(e.target.value)} 
              placeholder="Tìm kiếm văn bản pháp luật..." 
            />
            <Button type="submit" disabled={isLoading}>
              {isLoading ? '...' : 'Search'}
            </Button>
          </form>
          
          <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
            {['Luật', 'Nghị định', 'Pháp lệnh', 'Thông tư'].map(tag => (
              <Badge key={tag} className="cursor-pointer hover:bg-slate-200 transition">
                {tag}
              </Badge>
            ))}
          </div>

          <div className="mt-5 space-y-3 max-h-[600px] overflow-y-auto pr-1">
            {results.length > 0 ? (
              results.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedId(item.id)}
                  className={`w-full rounded-md border px-3 py-3 text-left text-sm transition ${
                    item.id === selectedId
                      ? 'border-slate-400 bg-slate-100 shadow-sm'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-900">{item.title}</span>
                    <Badge>{item.type}</Badge>
                  </div>
                  <div className="mt-1 text-xs text-slate-600">{item.ref} • {item.date}</div>
                  <p className="mt-2 line-clamp-2 text-slate-700 text-xs">{item.snippet}</p>
                </button>
              ))
            ) : (
              <p className="text-center text-slate-400 py-10">No documents found.</p>
            )}
          </div>
        </Panel>
      </div>

      {/* Right Content: Document Preview */}
      <Panel className="col-span-3 p-8 flex flex-col min-h-[700px]">
        {activeDoc ? (
          <>
            <div className="flex items-start justify-between border-b pb-6 mb-6">
              <div className="space-y-1">
                <p className="text-xs uppercase tracking-widest text-slate-500 font-bold">Văn bản pháp luật</p>
                <h2 className="text-2xl font-bold text-slate-900">{activeDoc.title}</h2>
                <div className="flex gap-2 mt-2">
                  <Badge className="bg-slate-900 text-white">{activeDoc.ref}</Badge>
                  <Badge className="border border-slate-300 bg-white text-slate-800">Năm ban hành: {activeDoc.date}</Badge>
                </div>
              </div>
              <Button className="shrink-0">Tải PDF</Button>
            </div>
            
            <div className="prose prose-slate max-w-none text-slate-800 leading-relaxed">
              <p className="whitespace-pre-line">
                {activeDoc.content || activeDoc.snippet}
              </p>
              
              {/* Extra placeholder text to simulate a long legal doc */}
              <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-400 text-sm text-blue-800 italic">
                Lưu ý: Đây là nội dung trích lục từ hệ thống cơ sở dữ liệu luật. Vui lòng đối chiếu với văn bản gốc để có độ chính xác tuyệt đối.
              </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <p>Chọn một văn bản từ danh sách bên trái để xem chi tiết.</p>
          </div>
        )}
      </Panel>
    </div>
  );
};