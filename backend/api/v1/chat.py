import logging

from fastapi import APIRouter, Depends, HTTPException, status

from models.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    """Dependency to get the shared chat service instance."""
    from main import chat_service
    return chat_service


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    """Send a message to Gemini and return its reply."""
    try:
        response_data = service.respond(payload.message)
    except Exception:
        logger.exception("Failed to handle chat request")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate response")

    return ChatResponse(
        reply=response_data["reply"],
        sources=response_data.get("sources", [])
    )
