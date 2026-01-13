import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from api.v1.chat import router as chat_router
from api.v1.lookup import router as lookup_router
from api.v1.auth import router as auth_router
from api.v1.sessions import router as sessions_router
from core.config import config
from core.logging import setup_logging
from core.database import init_db
from services.lookup_service import LawDocumentService
from services.chat_service import ChatService

logger = logging.getLogger(__name__)

app = FastAPI(title=config.APP_NAME)


# Global exception handlers
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors globally."""
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "A database error occurred. Please try again."},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc}")
    errors = exc.errors()
    # Format errors for user-friendly display
    messages = []
    for error in errors:
        field = ".".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        messages.append(f"{field}: {msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(messages)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors globally."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )

# Global service instances for reuse
lookup_service = LawDocumentService()
chat_service = ChatService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Debug middleware to log incoming request headers."""
    if "/auth/me" in request.url.path:
        auth_header = request.headers.get("Authorization", "NOT PRESENT")
        logger.info(f"[DEBUG] Request to {request.url.path}")
        logger.info(f"[DEBUG] Authorization header: {auth_header[:50] if auth_header != 'NOT PRESENT' else auth_header}...")
    response = await call_next(request)
    return response


@app.on_event("startup")
async def startup_event() -> None:
    setup_logging()
    init_db()  # Initialize database tables
    lookup_service.startup()
    chat_service.startup()


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router, prefix=config.API_STR)
app.include_router(sessions_router, prefix=config.API_STR)
app.include_router(chat_router, prefix=config.API_STR)
app.include_router(lookup_router, prefix=config.API_STR)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
