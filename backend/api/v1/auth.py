"""Authentication API endpoints."""

import logging
import re
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from api.deps import get_db, get_current_user
from core.security import create_access_token, PasswordTooLongError, MAX_PASSWORD_BYTES
from db.models.user import User
from models.auth import UserCreate, UserLogin, UserResponse, TokenResponse
from services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Validation constants
MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 50
MIN_PASSWORD_LENGTH = 6
USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


def validate_username(username: str) -> None:
    """Validate username format and length."""
    if len(username) < MIN_USERNAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username must be at least {MIN_USERNAME_LENGTH} characters",
        )
    if len(username) > MAX_USERNAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username must be at most {MAX_USERNAME_LENGTH} characters",
        )
    if not USERNAME_PATTERN.match(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username can only contain letters, numbers, and underscores",
        )


def validate_password(password: str) -> None:
    """Validate password length."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters",
        )
    # Check byte length for bcrypt
    password_bytes = len(password.encode('utf-8'))
    if password_bytes > MAX_PASSWORD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too long. Maximum {MAX_PASSWORD_BYTES} bytes allowed",
        )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user.

    Args:
        user_data: Registration data (username, password)
        db: Database session

    Returns:
        Created user info

    Raises:
        HTTPException: If validation fails or username already exists
    """
    logger.info(f"Registration attempt for username: {user_data.username}")

    # Validate input
    validate_username(user_data.username)
    validate_password(user_data.password)

    # Check if username already exists
    existing_user = auth_service.get_user_by_username(db, user_data.username)
    if existing_user:
        logger.warning(f"Registration failed: username '{user_data.username}' already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    try:
        # Create new user
        user = auth_service.create_user(db, user_data.username, user_data.password)
        logger.info(f"User registered successfully: {user.username} (id={user.id})")
        return user
    except PasswordTooLongError as e:
        logger.error(f"Password too long error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too long",
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error during registration: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account. Please try again.",
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db),
):
    """
    Authenticate user and return JWT token.

    Args:
        credentials: Login credentials (username, password)
        db: Database session

    Returns:
        JWT access token

    Raises:
        HTTPException: If credentials are invalid
    """
    logger.info(f"Login attempt for username: {credentials.username}")

    # Basic validation
    if not credentials.username or not credentials.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required",
        )

    try:
        user = auth_service.authenticate_user(
            db, credentials.username, credentials.password
        )

        if user is None:
            logger.warning(f"Login failed for username: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token with user ID as subject (must be string per JWT spec)
        access_token = create_access_token(data={"sub": str(user.id)})

        logger.info(f"User logged in successfully: {user.username}")

        return TokenResponse(access_token=access_token)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again.",
        )
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """
    Get current authenticated user info.

    Args:
        current_user: Current user from JWT token

    Returns:
        User info
    """
    return current_user
