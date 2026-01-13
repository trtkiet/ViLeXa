import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams, NavLink } from 'react-router-dom';
import { Panel, Badge, Input, Button } from '../components/ui';
import type { LawDocument, DocumentListResponse, TypesResponse } from '../types';
import { API_BASE_URL } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const API_URL = API_BASE_URL;

// Skeleton loading component
const DocumentSkeleton: React.FC = () => (
  <div className="animate-pulse space-y-3">
    {[1, 2, 3, 4, 5].map((i) => (
      <div key={i} className="rounded-md border border-slate-200 px-3 py-3">
        <div className="flex items-center justify-between">
          <div className="h-4 bg-slate-200 rounded w-3/4"></div>
          <div className="h-5 bg-slate-200 rounded w-16"></div>
        </div>
        <div className="mt-2 h-3 bg-slate-200 rounded w-1/3"></div>
        <div className="mt-2 h-3 bg-slate-200 rounded w-full"></div>
      </div>
    ))}
  </div>
);

// Content skeleton
const ContentSkeleton: React.FC = () => (
  <div className="animate-pulse">
    <div className="border-b pb-6 mb-6">
      <div className="h-3 bg-slate-200 rounded w-24 mb-2"></div>
      <div className="h-8 bg-slate-200 rounded w-3/4 mb-3"></div>
      <div className="flex gap-2">
        <div className="h-6 bg-slate-200 rounded w-20"></div>
        <div className="h-6 bg-slate-200 rounded w-32"></div>
      </div>
    </div>
    <div className="space-y-3">
      {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
        <div key={i} className="h-4 bg-slate-200 rounded" style={{ width: `${Math.random() * 40 + 60}%` }}></div>
      ))}
    </div>
  </div>
);

// Error message component
const ErrorMessage: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="flex flex-col items-center justify-center py-10 text-center">
    <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mb-3">
      <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </div>
    <p className="text-slate-600 mb-3">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 text-sm bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
      >
        Thử lại
      </button>
    )}
  </div>
);

export const LookupPage: React.FC = () => {
  const { user, logout } = useAuth();
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<LawDocument[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<LawDocument | null>(null);
  const [isListLoading, setIsListLoading] = useState(false);
  const [isDocLoading, setIsDocLoading] = useState(false);
  const [isTypesLoading, setIsTypesLoading] = useState(true);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [availableTypes, setAvailableTypes] = useState<string[]>([]);
  const [highlightArticle, setHighlightArticle] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [docError, setDocError] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 20;

  const contentRef = useRef<HTMLDivElement>(null);

  // Load available document types on mount
  useEffect(() => {
    const fetchTypes = async () => {
      setIsTypesLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/v1/documents/types`);
        if (response.ok) {
          const data: TypesResponse = await response.json();
          setAvailableTypes(data.types);
        }
      } catch (error) {
        console.error('Failed to fetch document types:', error);
      } finally {
        setIsTypesLoading(false);
      }
    };
    fetchTypes();
  }, []);

  // Handle citation navigation from ChatPage
  useEffect(() => {
    const docId = searchParams.get('id');
    const article = searchParams.get('article');

    if (docId) {
      setHighlightArticle(article);
      fetchDocumentById(docId);
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

  const fetchDocumentById = useCallback(async (docId: string) => {
    setIsDocLoading(true);
    setDocError(null);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents/${encodeURIComponent(docId)}`);
      if (response.ok) {
        const doc: LawDocument = await response.json();
        setSelectedDoc(doc);
        setResults([doc]);
      } else if (response.status === 404) {
        setDocError('Không tìm thấy văn bản này.');
      } else {
        setDocError('Không thể tải văn bản. Vui lòng thử lại.');
      }
    } catch (error) {
      console.error('Failed to fetch document:', error);
      setDocError('Lỗi kết nối. Vui lòng kiểm tra mạng và thử lại.');
    } finally {
      setIsDocLoading(false);
    }
  }, []);

  const fetchDocuments = useCallback(async (searchQuery?: string, docType?: string | null, page: number = 1) => {
    setIsListLoading(true);
    setListError(null);
    try {
      let url: string;
      const params = new URLSearchParams();

      if (docType) {
        params.append('type', docType);
      }
      params.append('page', page.toString());
      params.append('page_size', pageSize.toString());

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
        setCurrentPage(data.page);
        setTotalPages(data.total_pages);
        setTotalItems(data.total);

        if (data.items.length > 0 && !selectedDoc) {
          // Auto-select first result if no document is selected
          fetchFullDocument(data.items[0].id);
        }
      } else {
        setListError('Không thể tải danh sách văn bản.');
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      setListError('Lỗi kết nối. Vui lòng kiểm tra mạng và thử lại.');
    } finally {
      setIsListLoading(false);
    }
  }, [selectedDoc]);

  const fetchFullDocument = useCallback(async (docId: string) => {
    setIsDocLoading(true);
    setDocError(null);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents/${encodeURIComponent(docId)}`);
      if (response.ok) {
        const doc: LawDocument = await response.json();
        setSelectedDoc(doc);
      } else if (response.status === 404) {
        setDocError('Không tìm thấy văn bản này.');
      } else {
        setDocError('Không thể tải nội dung văn bản.');
      }
    } catch (error) {
      console.error('Failed to fetch document:', error);
      setDocError('Lỗi kết nối. Vui lòng thử lại.');
    } finally {
      setIsDocLoading(false);
    }
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setHighlightArticle(null);
    setCurrentPage(1);
    await fetchDocuments(query, selectedType, 1);
  };

  const handleTypeFilter = (type: string) => {
    const newType = selectedType === type ? null : type;
    setSelectedType(newType);
    setHighlightArticle(null);
    setCurrentPage(1);
    fetchDocuments(query, newType, 1);
  };

  const handleSelectDocument = (doc: LawDocument) => {
    setHighlightArticle(null);
    fetchFullDocument(doc.id);
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages && newPage !== currentPage) {
      setCurrentPage(newPage);
      fetchDocuments(query, selectedType, newPage);
    }
  };

  const retryFetchList = () => {
    fetchDocuments(query, selectedType, currentPage);
  };

  const retryFetchDoc = () => {
    if (selectedDoc?.id) {
      fetchFullDocument(selectedDoc.id);
    }
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
    <div className="h-screen flex flex-col bg-white">
      {/* Header */}
      <div className="h-14 border-b border-gray-100 flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-2">
          <span className="bg-gray-900 text-white px-2 py-0.5 rounded text-sm font-bold">VA</span>
          <span className="font-medium text-gray-900">Law Assistant</span>
        </div>
        <div className="flex items-center gap-4">
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
          {user && (
            <div className="flex items-center gap-2 pl-3 border-l border-gray-200">
              <div className="w-7 h-7 rounded-full bg-gray-900 text-white flex items-center justify-center text-xs font-medium">
                {user.username.charAt(0).toUpperCase()}
              </div>
              <button
                onClick={logout}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden p-6">
        <div className="grid grid-cols-5 gap-6 h-full">
          {/* Left Sidebar: Search & List */}
      <div className="col-span-2 flex flex-col gap-4">
        <Panel className="p-5">
          <form onSubmit={handleSearch} className="flex items-center gap-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Tìm kiếm văn bản pháp luật..."
              disabled={isListLoading}
            />
            <Button type="submit" disabled={isListLoading}>
              {isListLoading ? (
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : 'Tìm kiếm'}
            </Button>
          </form>

          {/* Document type filters */}
          <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
            {isTypesLoading ? (
              <div className="flex gap-2">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-6 w-16 bg-slate-200 rounded animate-pulse"></div>
                ))}
              </div>
            ) : availableTypes.length > 0 ? (
              availableTypes.slice(0, 8).map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => handleTypeFilter(tag)}
                  disabled={isListLoading}
                  className="p-0 border-0 bg-transparent disabled:opacity-50"
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
              ))
            ) : null}
          </div>

          {/* Results count */}
          {totalItems > 0 && !isListLoading && (
            <div className="mt-3 text-xs text-slate-500">
              Tìm thấy {totalItems} văn bản
              {selectedType && <span> loại "{selectedType}"</span>}
              {query && <span> cho "{query}"</span>}
            </div>
          )}

          {/* Document list */}
          <div className="mt-4 space-y-3 max-h-[500px] overflow-y-auto pr-1">
            {listError ? (
              <ErrorMessage message={listError} onRetry={retryFetchList} />
            ) : isListLoading ? (
              <DocumentSkeleton />
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
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-900 line-clamp-1">{item.title}</span>
                    <Badge className="shrink-0">{item.type}</Badge>
                  </div>
                  <div className="mt-1 text-xs text-slate-600">
                    {item.ref} • {item.date}
                  </div>
                  <p className="mt-2 line-clamp-2 text-slate-700 text-xs">{item.snippet}</p>
                </button>
              ))
            ) : (
              <div className="text-center text-slate-400 py-10">
                <svg className="w-12 h-12 mx-auto mb-3 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>{query || selectedType ? 'Không tìm thấy văn bản phù hợp.' : 'Tìm kiếm hoặc chọn loại văn bản để bắt đầu.'}</p>
              </div>
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && !listError && (
            <div className="mt-4 flex items-center justify-center gap-2">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1 || isListLoading}
                className="px-3 py-1 text-sm border rounded hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ‹
              </button>
              <div className="flex items-center gap-1">
                {/* Show first page */}
                {currentPage > 2 && (
                  <>
                    <button
                      onClick={() => handlePageChange(1)}
                      className="px-3 py-1 text-sm border rounded hover:bg-slate-100"
                    >
                      1
                    </button>
                    {currentPage > 3 && <span className="px-1 text-slate-400">...</span>}
                  </>
                )}
                {/* Show current page and neighbors */}
                {Array.from({ length: Math.min(3, totalPages) }, (_, i) => {
                  const page = Math.max(1, Math.min(currentPage - 1 + i, totalPages - 2 + i));
                  if (page < 1 || page > totalPages) return null;
                  if (currentPage > 2 && page === 1) return null;
                  if (currentPage < totalPages - 1 && page === totalPages) return null;
                  return (
                    <button
                      key={page}
                      onClick={() => handlePageChange(page)}
                      disabled={isListLoading}
                      className={`px-3 py-1 text-sm border rounded ${
                        page === currentPage
                          ? 'bg-slate-900 text-white border-slate-900'
                          : 'hover:bg-slate-100'
                      }`}
                    >
                      {page}
                    </button>
                  );
                })}
                {/* Show last page */}
                {currentPage < totalPages - 1 && (
                  <>
                    {currentPage < totalPages - 2 && <span className="px-1 text-slate-400">...</span>}
                    <button
                      onClick={() => handlePageChange(totalPages)}
                      className="px-3 py-1 text-sm border rounded hover:bg-slate-100"
                    >
                      {totalPages}
                    </button>
                  </>
                )}
              </div>
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages || isListLoading}
                className="px-3 py-1 text-sm border rounded hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ›
              </button>
            </div>
          )}
        </Panel>
      </div>

      {/* Right Content: Document Preview */}
      <Panel className="col-span-3 p-8 flex flex-col min-h-[700px]">
        {docError ? (
          <ErrorMessage message={docError} onRetry={retryFetchDoc} />
        ) : isDocLoading ? (
          <ContentSkeleton />
        ) : selectedDoc ? (
          <>
            <div className="flex items-start justify-between border-b pb-6 mb-6">
              <div className="space-y-1 flex-1 min-w-0">
                <p className="text-xs uppercase tracking-widest text-slate-500 font-bold">
                  Văn bản pháp luật
                </p>
                <h2 className="text-2xl font-bold text-slate-900">{selectedDoc.title}</h2>
                <div className="flex flex-wrap gap-2 mt-2">
                  <Badge className="bg-slate-900 text-white">{selectedDoc.ref}</Badge>
                  <Badge className="border border-slate-300 bg-white text-slate-800">
                    Năm ban hành: {selectedDoc.date}
                  </Badge>
                  {highlightArticle && (
                    <Badge className="bg-yellow-100 text-yellow-800 border border-yellow-300">
                      Đang xem: {highlightArticle}
                    </Badge>
                  )}
                </div>
              </div>
              <Button
                className="shrink-0 opacity-50 cursor-not-allowed"
                disabled
                title="Tính năng đang phát triển"
              >
                Tải PDF
              </Button>
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
            <svg className="w-16 h-16 mb-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p>Chọn một văn bản từ danh sách bên trái để xem chi tiết.</p>
          </div>
        )}
      </Panel>
        </div>
      </div>
    </div>
  );
};