import logging

from fastapi import APIRouter, Depends, HTTPException, status

from backend.models.chat import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    chat_service = ChatService()
    chat_service.startup()
    return chat_service


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    """Send a message to Gemini and return its reply."""
    try:
        reply = service.respond(payload.message)
    except Exception:
        logger.exception("Failed to handle chat request")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate response")

    return ChatResponse(reply=reply)
