"""MessageSource database model."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.message import Message


class MessageSource(Base):
    """Message source model for storing law citation sources."""

    __tablename__ = "message_sources"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chapter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    article: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    article_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    clause: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationship
    message: Mapped["Message"] = relationship("Message", back_populates="sources")
