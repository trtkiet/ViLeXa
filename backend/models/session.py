"""Pydantic schemas for chat sessions."""

from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Schema for creating a new session."""

    title: Optional[str] = Field(default="New Chat", max_length=255)


class SessionUpdate(BaseModel):
    """Schema for updating a session."""

    title: str = Field(..., min_length=1, max_length=255)


class SessionResponse(BaseModel):
    """Schema for session response."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """Schema for session list response."""

    sessions: List[SessionResponse]
    total: int


class SourceResponse(BaseModel):
    """Schema for message source."""

    # Vietnamese metadata fields
    document_id: Optional[str] = None
    document_type: Optional[str] = None
    document_title: Optional[str] = None
    phan: Optional[str] = None
    chuong: Optional[str] = None
    muc: Optional[str] = None
    dieu: Optional[str] = None

    # Legacy fields
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    article_title: Optional[str] = None
    clause: Optional[str] = None
    source_text: Optional[str] = None

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Schema for message response."""

    id: int
    role: str
    content: str
    timestamp: datetime
    sources: List[SourceResponse] = []

    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """Schema for session detail with messages."""

    messages: List[MessageResponse] = []
