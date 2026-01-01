"""
Pydantic models for law document lookup API.
"""

from typing import List, Optional
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Response model for a single law document."""
    id: str
    title: str
    type: str
    ref: str
    date: str
    snippet: str
    content: Optional[str] = None


class DocumentSummary(BaseModel):
    """Summary model for document list (without full content)."""
    id: str
    title: str
    type: str
    ref: str
    date: str
    snippet: str


class DocumentListResponse(BaseModel):
    """Response model for paginated document list."""
    items: List[DocumentSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class ArticleResponse(BaseModel):
    """Response model for a single article within a law."""
    article_number: str
    article_title: str
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None


class TypesResponse(BaseModel):
    """Response model for available document types."""
    types: List[str]
