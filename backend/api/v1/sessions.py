"""Chat session API endpoints."""

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from api.deps import get_db, get_current_user
from db.models.user import User
from models.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse,
    SessionDetailResponse,
    MessageResponse,
    SourceResponse,
)
from services.session_service import session_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def validate_session_id(session_id: str) -> None:
    """Validate that session_id is a valid UUID."""
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        )


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    data: SessionCreate = SessionCreate(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new chat session.

    Args:
        data: Session creation data (optional title)
        db: Database session
        current_user: Authenticated user

    Returns:
        Created session info
    """
    logger.info(f"Creating session for user {current_user.id}")

    session = session_service.create_session(
        db=db,
        user_id=current_user.id,
        title=data.title or "New Chat",
    )

    logger.info(f"Session created: {session.id}")
    return session


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all sessions for the current user.

    Args:
        db: Database session
        current_user: Authenticated user

    Returns:
        List of sessions ordered by updated_at desc
    """
    logger.info(f"Listing sessions for user {current_user.id}")

    sessions = session_service.get_sessions(db=db, user_id=current_user.id)
    total = session_service.count_sessions(db=db, user_id=current_user.id)

    return SessionListResponse(sessions=sessions, total=total)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get a session with all its messages.

    Args:
        session_id: Session UUID
        db: Database session
        current_user: Authenticated user

    Returns:
        Session with messages

    Raises:
        HTTPException: If session not found or not owned by user
    """
    validate_session_id(session_id)
    logger.info(f"Getting session {session_id} for user {current_user.id}")

    try:
        session = session_service.get_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error getting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load session",
        )

    # Get messages with sources
    messages = session_service.get_messages(db=db, session_id=session_id)

    # Build response
    message_responses = []
    for msg in messages:
        sources = [
            SourceResponse(
                # Vietnamese metadata
                document_id=s.document_id,
                document_type=s.document_type,
                document_title=s.document_title,
                phan=s.phan,
                chuong=s.chuong,
                muc=s.muc,
                dieu=s.dieu,
                # Legacy
                chapter=s.chapter,
                section=s.section,
                article=s.article,
                article_title=s.article_title,
                clause=s.clause,
                source_text=s.source_text,
            )
            for s in msg.sources
        ]
        message_responses.append(
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                sources=sources,
            )
        )

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=message_responses,
    )


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    data: SessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update a session's title.

    Args:
        session_id: Session UUID
        data: Update data (title)
        db: Database session
        current_user: Authenticated user

    Returns:
        Updated session

    Raises:
        HTTPException: If session not found or not owned by user
    """
    logger.info(f"Updating session {session_id} for user {current_user.id}")

    session = session_service.update_session_title(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
        title=data.title,
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    logger.info(f"Session updated: {session_id}")
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a session and all its messages.

    Args:
        session_id: Session UUID
        db: Database session
        current_user: Authenticated user

    Raises:
        HTTPException: If session not found or not owned by user
    """
    logger.info(f"Deleting session {session_id} for user {current_user.id}")

    deleted = session_service.delete_session(
        db=db,
        session_id=session_id,
        user_id=current_user.id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    logger.info(f"Session deleted: {session_id}")
