"""Security utilities for JWT and password hashing."""

from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import bcrypt
from jose import jwt, JWTError

from core.config import config

# Bcrypt has a maximum password length of 72 bytes
MAX_PASSWORD_BYTES = 72


class PasswordTooLongError(ValueError):
    """Raised when password exceeds bcrypt's 72 byte limit."""
    pass


def _check_password_length(password: str) -> None:
    """Check if password is within bcrypt's byte limit."""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"Password too long. Maximum {MAX_PASSWORD_BYTES} bytes allowed, "
            f"got {len(password_bytes)} bytes."
        )


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Raises:
        PasswordTooLongError: If password exceeds 72 bytes
    """
    _check_password_length(password)
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Returns False if password is too long (instead of raising).
    """
    try:
        _check_password_length(plain_password)
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except (PasswordTooLongError, ValueError):
        return False


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        config.JWT_SECRET_KEY,
        algorithm=config.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and validate a JWT access token."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Decoding token with key: {config.JWT_SECRET_KEY[:10]}... and algorithm: {config.JWT_ALGORITHM}")
        payload = jwt.decode(
            token,
            config.JWT_SECRET_KEY,
            algorithms=[config.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT decode error: {type(e).__name__}: {e}")
        return None
