import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Panel, Badge, Input, Button } from '../components/ui';
import type { LawDocument, DocumentListResponse, TypesResponse } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const LookupPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<LawDocument[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<LawDocument | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [availableTypes, setAvailableTypes] = useState<string[]>(['Luật', 'Nghị định', 'Pháp lệnh', 'Thông tư']);
  const [highlightArticle, setHighlightArticle] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Load available document types on mount
  useEffect(() => {
    const fetchTypes = async () => {
      try {
        const response = await fetch(`${API_URL}/api/v1/documents/types`);
        if (response.ok) {
          const data: TypesResponse = await response.json();
          setAvailableTypes(data.types);
        }
      } catch (error) {
        console.error('Failed to fetch document types:', error);
      }
    };
    fetchTypes();
  }, []);

  // Handle citation navigation from ChatPage
  useEffect(() => {
    const lawId = searchParams.get('law_id');
    const article = searchParams.get('article');

    if (lawId) {
      setHighlightArticle(article);
      fetchDocumentById(lawId);
    } else {
      // Load initial documents list
      fetchDocuments();
    }
  }, [searchParams]);

  // Scroll to highlighted article when content loads
  useEffect(() => {
    if (highlightArticle && contentRef.current && selectedDoc?.content) {
      // Find the article in the content and scroll to it
      setTimeout(() => {
        const articleMatch = selectedDoc.content?.match(new RegExp(`(${highlightArticle}[^\\n]*)`, 'i'));
        if (articleMatch && contentRef.current) {
          const content = contentRef.current;
          const text = content.innerText;
          const articleIndex = text.indexOf(articleMatch[1]);
          if (articleIndex !== -1) {
            // Approximate scroll position
            const lineHeight = 24;
            const linesBeforeArticle = text.substring(0, articleIndex).split('\n').length;
            content.scrollTop = linesBeforeArticle * lineHeight - 100;
          }
        }
      }, 100);
    }
  }, [highlightArticle, selectedDoc]);

  const fetchDocumentById = async (docId: string) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents/${encodeURIComponent(docId)}`);
      if (response.ok) {
        const doc: LawDocument = await response.json();
        setSelectedDoc(doc);
        setResults([doc]);
      } else {
        console.error('Document not found');
      }
    } catch (error) {
      console.error('Failed to fetch document:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDocuments = async (searchQuery?: string, docType?: string | null) => {
    setIsLoading(true);
    try {
      let url: string;
      const params = new URLSearchParams();
      
      if (docType) {
        params.append('type', docType);
      }
      params.append('page_size', '50');

      if (searchQuery && searchQuery.trim()) {
        params.append('q', searchQuery.trim());
        url = `${API_URL}/api/v1/documents/search?${params.toString()}`;
      } else {
        url = `${API_URL}/api/v1/documents?${params.toString()}`;
      }

      const response = await fetch(url);
      if (response.ok) {
        const data: DocumentListResponse = await response.json();
        setResults(data.items);
        if (data.items.length > 0 && !selectedDoc) {
          // Auto-select first result if no document is selected
          fetchFullDocument(data.items[0].id);
        }
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchFullDocument = async (docId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/v1/documents/${encodeURIComponent(docId)}`);
      if (response.ok) {
        const doc: LawDocument = await response.json();
        setSelectedDoc(doc);
      }
    } catch (error) {
      console.error('Failed to fetch document:', error);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setHighlightArticle(null);
    await fetchDocuments(query, selectedType);
  };

  const handleTypeFilter = (type: string) => {
    const newType = selectedType === type ? null : type;
    setSelectedType(newType);
    setHighlightArticle(null);
    fetchDocuments(query, newType);
  };

  const handleSelectDocument = (doc: LawDocument) => {
    setHighlightArticle(null);
    fetchFullDocument(doc.id);
  };

  // Highlight article in content
  const renderContent = (content: string) => {
    if (!highlightArticle) {
      return content;
    }

    // Split content by the highlighted article and wrap it
    const regex = new RegExp(`(${highlightArticle}[.:]?[^\\n]*)`, 'gi');
    const parts = content.split(regex);

    return parts.map((part, index) => {
      if (regex.test(part)) {
        return (
          <mark key={index} className="bg-yellow-200 px-1 rounded">
            {part}
          </mark>
        );
      }
      return part;
    });
  };

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
            {availableTypes.slice(0, 6).map((tag) => (
              <button
                key={tag}
                type="button"
                onClick={() => handleTypeFilter(tag)}
                className="p-0 border-0 bg-transparent"
              >
                <Badge
                  className={`cursor-pointer transition ${
                    selectedType === tag
                      ? 'bg-slate-900 text-white'
                      : 'hover:bg-slate-200'
                  }`}
                >
                  {tag}
                </Badge>
              </button>
            ))}
          </div>

          <div className="mt-5 space-y-3 max-h-[600px] overflow-y-auto pr-1">
            {isLoading && results.length === 0 ? (
              <p className="text-center text-slate-400 py-10">Đang tải...</p>
            ) : results.length > 0 ? (
              results.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleSelectDocument(item)}
                  className={`w-full rounded-md border px-3 py-3 text-left text-sm transition ${
                    item.id === selectedDoc?.id
                      ? 'border-slate-400 bg-slate-100 shadow-sm'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-900 line-clamp-1">{item.title}</span>
                    <Badge>{item.type}</Badge>
                  </div>
                  <div className="mt-1 text-xs text-slate-600">
                    {item.ref} • {item.date}
                  </div>
                  <p className="mt-2 line-clamp-2 text-slate-700 text-xs">{item.snippet}</p>
                </button>
              ))
            ) : (
              <p className="text-center text-slate-400 py-10">
                {query ? 'Không tìm thấy văn bản.' : 'Nhập từ khóa để tìm kiếm văn bản pháp luật.'}
              </p>
            )}
          </div>
        </Panel>
      </div>

      {/* Right Content: Document Preview */}
      <Panel className="col-span-3 p-8 flex flex-col min-h-[700px]">
        {selectedDoc ? (
          <>
            <div className="flex items-start justify-between border-b pb-6 mb-6">
              <div className="space-y-1">
                <p className="text-xs uppercase tracking-widest text-slate-500 font-bold">
                  Văn bản pháp luật
                </p>
                <h2 className="text-2xl font-bold text-slate-900">{selectedDoc.title}</h2>
                <div className="flex gap-2 mt-2">
                  <Badge className="bg-slate-900 text-white">{selectedDoc.ref}</Badge>
                  <Badge className="border border-slate-300 bg-white text-slate-800">
                    Năm ban hành: {selectedDoc.date}
                  </Badge>
                </div>
              </div>
              <Button className="shrink-0">Tải PDF</Button>
            </div>

            <div
              ref={contentRef}
              className="prose prose-slate max-w-none text-slate-800 leading-relaxed overflow-y-auto flex-1"
            >
              <p className="whitespace-pre-line">
                {selectedDoc.content
                  ? renderContent(selectedDoc.content)
                  : selectedDoc.snippet}
              </p>

              <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-400 text-sm text-blue-800 italic">
                Lưu ý: Đây là nội dung trích lục từ hệ thống cơ sở dữ liệu luật. Vui lòng đối
                chiếu với văn bản gốc để có độ chính xác tuyệt đối.
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