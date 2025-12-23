from typing import List, Dict, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[Dict[str, str]]] = None