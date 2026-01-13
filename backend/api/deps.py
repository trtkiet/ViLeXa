"""API dependencies for dependency injection."""

import logging
from typing import Generator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from core.database import SessionLocal
from core.security import decode_access_token
from db.models.user import User

logger = logging.getLogger(__name__)

# OAuth2 scheme for JWT token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login", auto_error=False
)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Get the current authenticated user.
    Raises HTTPException if token is invalid or user not found.
    """
    logger.info(f"get_current_user called with token: {token[:20]}..." if token else "get_current_user called with no token")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    logger.info(f"Decoded payload: {payload}")

    if payload is None:
        logger.warning("Token decode failed - payload is None")
        raise credentials_exception

    user_id_str: Optional[str] = payload.get("sub")
    logger.info(f"Extracted user_id: {user_id_str} (type: {type(user_id_str)})")

    if user_id_str is None:
        logger.warning("user_id is None in payload")
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        logger.warning(f"Invalid user_id format: {user_id_str}")
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning(f"User not found with id: {user_id}")
        raise credentials_exception

    logger.info(f"User authenticated: {user.username}")
    return user


def get_current_user_optional(
    db: Session = Depends(get_db),
    token: Optional[str] = Depends(oauth2_scheme_optional),
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.
    Does not raise exception for missing/invalid token.
    """
    if token is None:
        return None

    payload = decode_access_token(token)
    if payload is None:
        return None

    user_id_str: Optional[str] = payload.get("sub")
    if user_id_str is None:
        return None

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return None

    return db.query(User).filter(User.id == user_id).first()
