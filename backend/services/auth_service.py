"""Authentication service for user management."""

from typing import Optional

from sqlalchemy.orm import Session

from core.security import hash_password, verify_password
from db.models.user import User


class AuthService:
    """Service for handling user authentication operations."""

    def create_user(self, db: Session, username: str, password: str) -> User:
        """
        Create a new user with hashed password.

        Args:
            db: Database session
            username: User's username
            password: Plain text password

        Returns:
            Created User object
        """
        hashed = hash_password(password)
        user = User(username=username, password_hash=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def authenticate_user(
        self, db: Session, username: str, password: str
    ) -> Optional[User]:
        """
        Authenticate a user by username and password.

        Args:
            db: Database session
            username: User's username
            password: Plain text password

        Returns:
            User object if credentials are valid, None otherwise
        """
        user = self.get_user_by_username(db, username)
        if user is None:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def get_user_by_id(self, db: Session, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get a user by username."""
        return db.query(User).filter(User.username == username).first()


# Global instance
auth_service = AuthService()
