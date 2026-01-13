from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str
    session_id: Optional[str] = None  # If None, creates new session


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    reply: str
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str  # Always return session_id (created or existing)