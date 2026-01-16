"""Chat API endpoint with authentication and message persistence."""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from api.deps import get_db, get_current_user
from db.models.user import User
from models.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService
from services.session_service import session_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    """Dependency to get the shared chat service instance."""
    from main import chat_service

    return chat_service


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Send a message to Gemini and return its reply.
    Requires authentication. Messages are persisted to the database.

    Args:
        payload: Chat request with message and optional session_id
        db: Database session
        current_user: Authenticated user
        service: Chat service instance

    Returns:
        ChatResponse with reply, sources, and session_id
    """
    session_id = payload.session_id

    # Create new session if not provided
    if not session_id:
        logger.info(f"Creating new session for user {current_user.id}")
        chat_session = session_service.create_session(
            db=db,
            user_id=current_user.id,
            title="New Chat",
        )
        session_id = chat_session.id
        is_new_session = True
    else:
        # Verify session ownership
        chat_session = session_service.get_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )
        if chat_session is None:
            logger.warning(f"Session {session_id} not found for user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        is_new_session = False

    logger.info(
        f"Chat request: user={current_user.id}, session={session_id}, new={is_new_session}"
    )

    # Load existing messages from database into LangChain history
    # This ensures continuity even if the server restarted
    existing_messages = session_service.get_messages(db=db, session_id=session_id)

    # Populate the in-memory history with DB messages
    history = ChatMessageHistory()
    for msg in existing_messages:
        if msg.role == "user":
            history.add_message(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            history.add_message(AIMessage(content=msg.content))

    # Inject the loaded history into the service's session_store
    service.session_store[session_id] = history

    try:
        # Call the RAG pipeline
        response_data = service.respond(payload.message, session_id=session_id)
    except Exception:
        logger.exception("Failed to handle chat request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response",
        )

    # Save user message to database
    session_service.add_message(
        db=db,
        session_id=session_id,
        role="user",
        content=payload.message,
        sources=None,
    )

    # Save AI response to database with sources
    sources = response_data.get("sources", [])
    session_service.add_message(
        db=db,
        session_id=session_id,
        role="ai",
        content=response_data["answer"],
        sources=sources,
    )

    # Update session timestamp
    session_service.update_session_timestamp(db=db, session_id=session_id)

    # Auto-title session from first message if it's a new session
    if is_new_session:
        session_service.auto_title_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            first_message=payload.message,
        )

    return ChatResponse(
        reply=response_data["answer"],
        sources=sources,
        session_id=session_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Send a message and stream the response via Server-Sent Events.

    SSE Events:
    - data: {"type": "session", "session_id": "..."} - Session info (first event)
    - data: {"type": "sources", "sources": [...]} - Source documents
    - data: {"type": "token", "token": "..."} - Generated tokens
    - data: {"type": "done", "answer": "..."} - Complete response
    - data: {"type": "error", "error": "..."} - Error message
    """
    session_id = payload.session_id
    is_new_session = False

    # Create new session if not provided
    if not session_id:
        logger.info(f"[Stream] Creating new session for user {current_user.id}")
        chat_session = session_service.create_session(
            db=db,
            user_id=current_user.id,
            title="New Chat",
        )
        session_id = chat_session.id
        is_new_session = True
    else:
        # Verify session ownership
        chat_session = session_service.get_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )
        if chat_session is None:
            logger.warning(
                f"[Stream] Session {session_id} not found for user {current_user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

    logger.info(
        f"[Stream] Chat request: user={current_user.id}, session={session_id}, new={is_new_session}"
    )

    # Load existing messages from database into LangChain history
    existing_messages = session_service.get_messages(db=db, session_id=session_id)

    # Populate the in-memory history with DB messages
    history = ChatMessageHistory()
    for msg in existing_messages:
        if msg.role == "user":
            history.add_message(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            history.add_message(AIMessage(content=msg.content))

    # Inject the loaded history into the service's session_store
    service.session_store[session_id] = history

    # Store db session and other context for the generator
    user_message = payload.message

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        # Send session info first
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        full_answer = ""
        sources = []

        try:
            for chunk in service.stream_respond(user_message, session_id=session_id):
                chunk_type = chunk.get("type")

                if chunk_type == "sources":
                    sources = chunk.get("sources", [])
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk_type == "token":
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk_type == "done":
                    full_answer = chunk.get("answer", "")
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk_type == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
                    return

            # Save messages to database after streaming completes
            session_service.add_message(
                db=db,
                session_id=session_id,
                role="user",
                content=user_message,
                sources=None,
            )

            session_service.add_message(
                db=db,
                session_id=session_id,
                role="ai",
                content=full_answer,
                sources=sources,
            )

            # Update session timestamp
            session_service.update_session_timestamp(db=db, session_id=session_id)

            # Auto-title session from first message if it's a new session
            if is_new_session:
                session_service.auto_title_session(
                    db=db,
                    session_id=session_id,
                    user_id=current_user.id,
                    first_message=user_message,
                )

        except Exception as e:
            logger.exception("[Stream] Failed to handle chat request")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
