"""Session service for chat session management."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from sqlalchemy.orm import Session

from db.models.chat_session import ChatSession
from db.models.message import Message
from db.models.message_source import MessageSource


class SessionService:
    """Service for handling chat session operations."""

    def create_session(
        self,
        db: Session,
        user_id: int,
        title: str = "New Chat",
    ) -> ChatSession:
        """
        Create a new chat session.

        Args:
            db: Database session
            user_id: Owner user ID
            title: Session title

        Returns:
            Created ChatSession object
        """
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def get_sessions(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 50,
    ) -> List[ChatSession]:
        """
        Get all sessions for a user, ordered by updated_at desc.

        Args:
            db: Database session
            user_id: Owner user ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of ChatSession objects
        """
        return (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count_sessions(self, db: Session, user_id: int) -> int:
        """Count total sessions for a user."""
        return db.query(ChatSession).filter(ChatSession.user_id == user_id).count()

    def get_session(
        self,
        db: Session,
        session_id: str,
        user_id: int,
    ) -> Optional[ChatSession]:
        """
        Get a session by ID, verifying ownership.

        Args:
            db: Database session
            session_id: Session UUID
            user_id: Owner user ID

        Returns:
            ChatSession if found and owned by user, None otherwise
        """
        return (
            db.query(ChatSession)
            .filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
            .first()
        )

    def update_session_title(
        self,
        db: Session,
        session_id: str,
        user_id: int,
        title: str,
    ) -> Optional[ChatSession]:
        """
        Update a session's title.

        Args:
            db: Database session
            session_id: Session UUID
            user_id: Owner user ID
            title: New title

        Returns:
            Updated ChatSession if found, None otherwise
        """
        session = self.get_session(db, session_id, user_id)
        if session is None:
            return None

        session.title = title
        session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(session)
        return session

    def delete_session(
        self,
        db: Session,
        session_id: str,
        user_id: int,
    ) -> bool:
        """
        Delete a session (cascade deletes messages and sources).

        Args:
            db: Database session
            session_id: Session UUID
            user_id: Owner user ID

        Returns:
            True if deleted, False if not found
        """
        session = self.get_session(db, session_id, user_id)
        if session is None:
            return False

        db.delete(session)
        db.commit()
        return True

    def add_message(
        self,
        db: Session,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """
        Add a message to a session.

        Args:
            db: Database session
            session_id: Session UUID
            role: Message role ('user' or 'ai')
            content: Message content
            sources: Optional list of source dictionaries for AI messages

        Returns:
            Created Message object
        """
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
        )
        db.add(message)
        db.flush()  # Get message.id without committing

        # Add sources if provided (typically for AI messages)
        if sources:
            for source_data in sources:
                source = MessageSource(
                    message_id=message.id,
                    chapter=source_data.get("chapter"),
                    section=source_data.get("section"),
                    article=source_data.get("article"),
                    article_title=source_data.get("article_title"),
                    clause=source_data.get("clause"),
                    source_text=source_data.get("source_text"),
                )
                db.add(source)

        db.commit()
        db.refresh(message)
        return message

    def get_messages(
        self,
        db: Session,
        session_id: str,
    ) -> List[Message]:
        """
        Get all messages for a session.

        Args:
            db: Database session
            session_id: Session UUID

        Returns:
            List of Message objects with sources loaded
        """
        return (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp)
            .all()
        )

    def update_session_timestamp(
        self,
        db: Session,
        session_id: str,
    ) -> None:
        """Update a session's updated_at timestamp."""
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.utcnow()
            db.commit()

    def auto_title_session(
        self,
        db: Session,
        session_id: str,
        user_id: int,
        first_message: str,
    ) -> None:
        """
        Auto-generate session title from first message.
        Only updates if current title is 'New Chat'.
        """
        session = self.get_session(db, session_id, user_id)
        if session and session.title == "New Chat":
            # Truncate to ~50 chars, try to break at word boundary
            title = first_message[:50]
            if len(first_message) > 50:
                # Try to break at last space
                last_space = title.rfind(" ")
                if last_space > 20:
                    title = title[:last_space]
                title += "..."
            session.title = title
            db.commit()


# Global instance
session_service = SessionService()
