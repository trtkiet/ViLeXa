import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useSearchParams, NavLink } from 'react-router-dom';
import { Badge, Input, Button } from '../components/ui';
import type { LawDocument, DocumentListResponse, TypesResponse } from '../types';
import { API_BASE_URL } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

const API_URL = API_BASE_URL;

// Table of Contents item
interface TocItem {
  id: string;
  title: string;
  level: number;
  position: number;
}

// Extract table of contents from document content
function extractToc(content: string): TocItem[] {
  const items: TocItem[] = [];
  const lines = content.split('\n');

  // Patterns for Vietnamese legal document structure
  const patterns = [
    { regex: /^(PHẦN\s+[IVXLCDM\d]+)[.:\s]*(.*)$/i, level: 1 },
    { regex: /^(Chương\s+[IVXLCDM\d]+)[.:\s]*(.*)$/i, level: 2 },
    { regex: /^(Mục\s+\d+)[.:\s]*(.*)$/i, level: 3 },
    { regex: /^(Điều\s+\d+)[.:\s]*(.*)$/i, level: 4 },
  ];

  let charPosition = 0;
  lines.forEach((line) => {
    const trimmedLine = line.trim();
    for (const { regex, level } of patterns) {
      const match = trimmedLine.match(regex);
      if (match) {
        items.push({
          id: `toc-${items.length}`,
          title: match[2] ? `${match[1]}: ${match[2].trim()}` : match[1],
          level,
          position: charPosition,
        });
        break;
      }
    }
    charPosition += line.length + 1;
  });

  return items;
}

// Skeleton loading component
const DocumentSkeleton: React.FC = () => (
  <div className="animate-pulse space-y-2">
    {[1, 2, 3, 4, 5, 6].map((i) => (
      <div key={i} className="rounded-lg border border-slate-100 p-3 bg-white">
        <div className="h-4 bg-slate-100 rounded w-4/5 mb-2"></div>
        <div className="h-3 bg-slate-100 rounded w-1/3 mb-2"></div>
        <div className="h-3 bg-slate-100 rounded w-full"></div>
      </div>
    ))}
  </div>
);

// Content skeleton
const ContentSkeleton: React.FC = () => (
  <div className="animate-pulse p-6">
    <div className="h-3 bg-slate-100 rounded w-24 mb-3"></div>
    <div className="h-7 bg-slate-100 rounded w-3/4 mb-4"></div>
    <div className="flex gap-2 mb-6">
      <div className="h-6 bg-slate-100 rounded w-24"></div>
      <div className="h-6 bg-slate-100 rounded w-32"></div>
    </div>
    <div className="space-y-3">
      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
        <div key={i} className="h-4 bg-slate-100 rounded" style={{ width: `${Math.random() * 30 + 70}%` }}></div>
      ))}
    </div>
  </div>
);

// Error message component
const ErrorMessage: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="flex flex-col items-center justify-center py-12 text-center">
    <div className="w-14 h-14 rounded-full bg-red-50 flex items-center justify-center mb-4">
      <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </div>
    <p className="text-slate-600 mb-4">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 text-sm bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors"
      >
        Thử lại
      </button>
    )}
  </div>
);

// Back to top button
const BackToTopButton: React.FC<{ onClick: () => void; visible: boolean }> = ({ onClick, visible }) => (
  <button
    onClick={onClick}
    className={`fixed bottom-6 right-6 w-10 h-10 bg-slate-900 text-white rounded-full shadow-lg flex items-center justify-center transition-all duration-300 hover:bg-slate-800 hover:scale-110 z-50 ${
      visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'
    }`}
    aria-label="Cuộn lên đầu"
  >
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
    </svg>
  </button>
);

// Reading progress bar
const ReadingProgress: React.FC<{ progress: number }> = ({ progress }) => (
  <div className="h-0.5 bg-slate-100 w-full">
    <div
      className="h-full bg-slate-900 transition-all duration-150 ease-out"
      style={{ width: `${progress}%` }}
    />
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
  const [readingProgress, setReadingProgress] = useState(0);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [showToc, setShowToc] = useState(true);
  const [isExporting, setIsExporting] = useState(false);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const pageSize = 20;

  const contentRef = useRef<HTMLDivElement>(null);
  const contentWrapperRef = useRef<HTMLDivElement>(null);

  // Extract TOC from content
  const toc = useMemo(() => {
    if (!selectedDoc?.content) return [];
    return extractToc(selectedDoc.content);
  }, [selectedDoc?.content]);

  // Handle scroll for progress and back to top
  const handleContentScroll = useCallback(() => {
    if (!contentWrapperRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = contentWrapperRef.current;
    const progress = scrollHeight > clientHeight
      ? (scrollTop / (scrollHeight - clientHeight)) * 100
      : 0;

    setReadingProgress(Math.min(100, Math.max(0, progress)));
    setShowBackToTop(scrollTop > 300);
  }, []);

  const scrollToTop = useCallback(() => {
    contentWrapperRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const scrollToTocItem = useCallback((item: TocItem) => {
    const wrapperEl = contentWrapperRef.current;
    if (!wrapperEl) return;

    // Find element by TOC id
    const targetElement = document.getElementById(item.id);
    if (!targetElement) return;

    // Calculate the target scroll position
    const wrapperRect = wrapperEl.getBoundingClientRect();
    const elementRect = targetElement.getBoundingClientRect();
    const currentScrollTop = wrapperEl.scrollTop;
    
    // Calculate where to scroll: element position relative to container + current scroll - offset for header
    const targetScrollTop = currentScrollTop + (elementRect.top - wrapperRect.top) - 80;
    
    wrapperEl.scrollTo({
      top: Math.max(0, targetScrollTop),
      behavior: 'smooth'
    });

    // Wait for scroll to finish, then flash highlight effect
    const scrollDistance = Math.abs(targetScrollTop - currentScrollTop);
    const scrollDuration = Math.min(500, scrollDistance * 0.5); // Estimate scroll duration
    
    setTimeout(() => {
      targetElement.classList.add('bg-yellow-200', 'transition-colors', 'duration-300');
      setTimeout(() => {
        targetElement.classList.remove('bg-yellow-200');
        targetElement.classList.add('bg-transparent');
        setTimeout(() => {
          targetElement.classList.remove('transition-colors', 'duration-300', 'bg-transparent');
        }, 300);
      }, 800);
    }, scrollDuration);
  }, []);

  // PDF Export function using browser print
  const exportToPdf = useCallback(() => {
    if (!selectedDoc) return;

    setIsExporting(true);

    try {
      // Create a new window for printing
      const printWindow = window.open('', '_blank');
      if (!printWindow) {
        alert('Vui lòng cho phép popup để xuất PDF');
        setIsExporting(false);
        return;
      }

      const content = selectedDoc.content || selectedDoc.snippet || '';

      // Write the print-friendly HTML
      printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>${selectedDoc.ref} - ${selectedDoc.title}</title>
          <style>
            * {
              margin: 0;
              padding: 0;
              box-sizing: border-box;
            }
            body {
              font-family: 'Times New Roman', serif;
              font-size: 12pt;
              line-height: 1.6;
              padding: 20mm;
              color: #000;
            }
            .header {
              text-align: center;
              margin-bottom: 20pt;
              padding-bottom: 15pt;
              border-bottom: 1px solid #ccc;
            }
            .doc-type {
              font-size: 10pt;
              color: #666;
              text-transform: uppercase;
              letter-spacing: 2px;
              margin-bottom: 10pt;
            }
            .title {
              font-size: 16pt;
              font-weight: bold;
              margin-bottom: 15pt;
            }
            .meta {
              font-size: 10pt;
              color: #444;
            }
            .meta span {
              margin: 0 10pt;
            }
            .content {
              white-space: pre-line;
              text-align: justify;
            }
            .footer {
              margin-top: 30pt;
              padding-top: 15pt;
              border-top: 1px solid #ccc;
              font-size: 9pt;
              color: #666;
              text-align: center;
            }
            @media print {
              body { padding: 15mm; }
              @page { margin: 15mm; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="doc-type">Văn bản pháp luật</div>
            <div class="title">${selectedDoc.title}</div>
            <div class="meta">
              <span><strong>Số hiệu:</strong> ${selectedDoc.ref}</span>
              <span><strong>Năm:</strong> ${selectedDoc.date}</span>
              <span><strong>Loại:</strong> ${selectedDoc.type}</span>
            </div>
          </div>
          <div class="content">${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
          <div class="footer">
            Xuất từ Vietnamese Law Assistant - ${new Date().toLocaleDateString('vi-VN')}
          </div>
        </body>
        </html>
      `);

      printWindow.document.close();

      // Wait for content to load then print
      printWindow.onload = () => {
        printWindow.focus();
        printWindow.print();
        // Close after a delay to allow print dialog
        setTimeout(() => {
          printWindow.close();
        }, 1000);
      };

    } catch (error) {
      console.error('PDF export failed:', error);
      alert('Không thể xuất PDF. Vui lòng thử lại.');
    } finally {
      setIsExporting(false);
    }
  }, [selectedDoc]);

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
      fetchDocuments();
    }
  }, [searchParams]);

  // Scroll to highlighted article when content loads
  useEffect(() => {
    if (highlightArticle && contentRef.current && selectedDoc?.content) {
      setTimeout(() => {
        // Find element containing the article text in the DOM
        const contentEl = contentRef.current;
        if (!contentEl) return;
        
        // Search through all text nodes to find the article
        const walker = document.createTreeWalker(
          contentEl,
          NodeFilter.SHOW_TEXT,
          null
        );
        
        let node;
        while ((node = walker.nextNode())) {
          if (node.textContent && node.textContent.includes(highlightArticle)) {
            // Found the node, scroll its parent element into view
            const parentElement = node.parentElement;
            if (parentElement) {
              parentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
              // Add highlight effect
              parentElement.style.backgroundColor = '#fef08a';
              setTimeout(() => {
                parentElement.style.backgroundColor = '';
              }, 2000);
            }
            break;
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
    setReadingProgress(0);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents/${encodeURIComponent(docId)}`);
      if (response.ok) {
        const doc: LawDocument = await response.json();
        setSelectedDoc(doc);
        contentWrapperRef.current?.scrollTo({ top: 0 });
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

  // Render content with TOC anchors and optional highlighting
  const renderContent = (content: string) => {
    // Patterns for Vietnamese legal document structure (same as extractToc)
    const tocPatterns = [
      /^(PHẦN\s+[IVXLCDM\d]+)[.:\s]*(.*)/i,
      /^(Chương\s+[IVXLCDM\d]+)[.:\s]*(.*)/i,
      /^(Mục\s+\d+)[.:\s]*(.*)/i,
      /^(Điều\s+\d+)[.:\s]*(.*)/i,
    ];

    const lines = content.split('\n');
    let tocIndex = 0;
    
    return lines.map((line, lineIndex) => {
      const trimmedLine = line.trim();
      let isTocItem = false;
      let currentTocId = '';
      
      // Check if this line matches any TOC pattern
      for (const pattern of tocPatterns) {
        if (pattern.test(trimmedLine)) {
          isTocItem = true;
          currentTocId = `toc-${tocIndex}`;
          tocIndex++;
          break;
        }
      }
      
      // Check if this line should be highlighted
      const shouldHighlight = highlightArticle && 
        new RegExp(`${highlightArticle}[.:]?`, 'i').test(line);
      
      if (isTocItem) {
        return (
          <span key={lineIndex}>
            <span id={currentTocId} className={`scroll-mt-24 ${shouldHighlight ? 'bg-yellow-200 px-1 rounded' : ''}`}>
              {line}
            </span>
            {'\n'}
          </span>
        );
      }
      
      if (shouldHighlight) {
        return (
          <span key={lineIndex}>
            <mark className="bg-yellow-200 px-1 rounded">{line}</mark>
            {'\n'}
          </span>
        );
      }
      
      return lineIndex < lines.length - 1 ? line + '\n' : line;
    });
  };

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 shrink-0 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="bg-slate-900 text-white px-2.5 py-1 rounded-lg text-sm font-bold tracking-wide">VLA</span>
          <span className="font-semibold text-slate-800">Law Assistant</span>
        </div>
        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
            <NavLink
              to="/"
              className={({isActive}) =>
                `px-4 py-1.5 text-sm rounded-md transition-all duration-200 ${
                  isActive
                    ? 'bg-white font-medium shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`
              }
            >
              Chat
            </NavLink>
            <NavLink
              to="/lookup"
              className={({isActive}) =>
                `px-4 py-1.5 text-sm rounded-md transition-all duration-200 ${
                  isActive
                    ? 'bg-white font-medium shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`
              }
            >
              Tra cứu
            </NavLink>
          </nav>
          {user && (
            <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-700 to-slate-900 text-white flex items-center justify-center text-sm font-medium">
                {user.username.charAt(0).toUpperCase()}
              </div>
              <button
                onClick={logout}
                className="text-sm text-slate-500 hover:text-slate-800 transition-colors"
              >
                Đăng xuất
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Left Sidebar: Search & List */}
        <aside className="w-96 bg-white border-r border-slate-200 flex flex-col shrink-0">
          {/* Search */}
          <div className="p-4 border-b border-slate-100">
            <form onSubmit={handleSearch} className="flex gap-2">
              <div className="relative flex-1">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Tìm kiếm văn bản..."
                  disabled={isListLoading}
                  className="pl-9"
                />
              </div>
              <Button type="submit" disabled={isListLoading} className="shrink-0">
                {isListLoading ? (
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : 'Tìm'}
              </Button>
            </form>

            {/* Document type filters */}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {isTypesLoading ? (
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-6 w-16 bg-slate-100 rounded-full animate-pulse"></div>
                  ))}
                </div>
              ) : availableTypes.length > 0 ? (
                availableTypes.slice(0, 8).map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => handleTypeFilter(tag)}
                    disabled={isListLoading}
                    className={`px-2.5 py-1 text-xs rounded-full transition-all duration-200 disabled:opacity-50 ${
                      selectedType === tag
                        ? 'bg-slate-900 text-white'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {tag}
                  </button>
                ))
              ) : null}
            </div>
          </div>

          {/* Results info */}
          {totalItems > 0 && !isListLoading && (
            <div className="px-4 py-2 text-xs text-slate-500 bg-slate-50 border-b border-slate-100">
              <span className="font-medium text-slate-700">{totalItems.toLocaleString()}</span> văn bản
              {selectedType && <span className="text-slate-400"> • {selectedType}</span>}
              {query && <span className="text-slate-400"> • "{query}"</span>}
            </div>
          )}

          {/* Document list */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-3 space-y-2">
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
                    className={`w-full rounded-lg border p-3 text-left transition-all duration-200 group ${
                      item.id === selectedDoc?.id
                        ? 'border-slate-300 bg-slate-100 shadow-sm'
                        : 'border-transparent bg-white hover:border-slate-200 hover:shadow-sm'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="font-medium text-slate-900 text-sm leading-snug line-clamp-2 group-hover:text-slate-700">
                        {item.title}
                      </h3>
                      <Badge className="shrink-0 text-[10px]">{item.type}</Badge>
                    </div>
                    <div className="mt-1.5 flex items-center gap-2 text-xs text-slate-500">
                      <span className="font-mono">{item.ref}</span>
                      <span className="w-1 h-1 rounded-full bg-slate-300"></span>
                      <span>{item.date}</span>
                    </div>
                    <p className="mt-2 text-xs text-slate-600 line-clamp-2 leading-relaxed">{item.snippet}</p>
                  </button>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="text-slate-500 text-sm">
                    {query || selectedType ? 'Không tìm thấy văn bản phù hợp' : 'Nhập từ khóa để tìm kiếm'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && !listError && (
            <div className="p-3 border-t border-slate-100 bg-white">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1 || isListLoading}
                  className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <span className="text-sm text-slate-600">
                  Trang <span className="font-medium">{currentPage}</span> / {totalPages}
                </span>
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages || isListLoading}
                  className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          )}
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 flex overflow-hidden">
          {/* Document Content */}
          <div className="flex-1 flex flex-col bg-white">
            {/* Reading progress */}
            {selectedDoc && !isDocLoading && (
              <ReadingProgress progress={readingProgress} />
            )}

            {/* Content */}
            <div
              ref={contentWrapperRef}
              onScroll={handleContentScroll}
              className="flex-1 overflow-y-auto"
            >
              {docError ? (
                <ErrorMessage message={docError} onRetry={retryFetchDoc} />
              ) : isDocLoading ? (
                <ContentSkeleton />
              ) : selectedDoc ? (
                <div className="max-w-4xl mx-auto p-8">
                  {/* Document header */}
                  <div className="mb-8 pb-6 border-b border-slate-200">
                    <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
                      Văn bản pháp luật
                    </p>
                    <h1 className="text-2xl font-bold text-slate-900 leading-tight mb-4">
                      {selectedDoc.title}
                    </h1>
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge className="bg-slate-900 text-white font-mono">{selectedDoc.ref}</Badge>
                      <Badge className="bg-slate-100 text-slate-700">
                        {selectedDoc.date}
                      </Badge>
                      <Badge className="bg-blue-50 text-blue-700 border border-blue-200">
                        {selectedDoc.type}
                      </Badge>
                      {highlightArticle && (
                        <Badge className="bg-yellow-100 text-yellow-800 border border-yellow-300">
                          {highlightArticle}
                        </Badge>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="mt-4 flex items-center gap-2">
                      <Button
                        onClick={exportToPdf}
                        disabled={isExporting}
                        className="flex items-center gap-2"
                      >
                        {isExporting ? (
                          <>
                            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Đang xuất...
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Xuất PDF
                          </>
                        )}
                      </Button>

                      {toc.length > 0 && (
                        <button
                          onClick={() => setShowToc(!showToc)}
                          className={`px-3 py-2 text-sm rounded-lg border transition-colors ${
                            showToc
                              ? 'bg-slate-100 border-slate-200 text-slate-700'
                              : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                          }`}
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                          </svg>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Document content */}
                  <div
                    ref={contentRef}
                    className="prose prose-slate max-w-none text-slate-700 leading-relaxed"
                  >
                    <div className="whitespace-pre-line text-[15px] leading-7">
                      {selectedDoc.content
                        ? renderContent(selectedDoc.content)
                        : selectedDoc.snippet}
                    </div>

                    <div className="mt-8 p-4 bg-amber-50 border-l-4 border-amber-400 text-sm text-amber-800 rounded-r-lg">
                      <strong>Lưu ý:</strong> Đây là nội dung trích lục từ hệ thống cơ sở dữ liệu.
                      Vui lòng đối chiếu với văn bản gốc để có độ chính xác tuyệt đối.
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center p-8">
                  <div className="w-20 h-20 rounded-full bg-slate-100 flex items-center justify-center mb-6">
                    <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-slate-700 mb-2">Chọn văn bản để xem</h3>
                  <p className="text-slate-500 text-sm max-w-sm">
                    Tìm kiếm hoặc chọn một văn bản từ danh sách bên trái để xem nội dung chi tiết
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Table of Contents Sidebar */}
          {selectedDoc && toc.length > 0 && showToc && !isDocLoading && (
            <aside className="w-64 bg-slate-50 border-l border-slate-200 overflow-y-auto shrink-0">
              <div className="p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                  Mục lục
                </h3>
                <nav className="space-y-1">
                  {toc.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => scrollToTocItem(item)}
                      className={`w-full text-left text-sm py-1.5 px-2 rounded transition-colors hover:bg-slate-200 text-slate-600 hover:text-slate-900 ${
                        item.level === 1 ? 'font-semibold' :
                        item.level === 2 ? 'pl-4' :
                        item.level === 3 ? 'pl-6 text-xs' :
                        'pl-8 text-xs'
                      }`}
                    >
                      {item.title}
                    </button>
                  ))}
                </nav>
              </div>
            </aside>
          )}
        </main>
      </div>

      {/* Back to top button */}
      <BackToTopButton onClick={scrollToTop} visible={showBackToTop} />
    </div>
  );
};
