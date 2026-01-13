from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    APP_NAME: str = "Vietnamese Law API"
    API_STR: str = "/api/v1"
    SERVER_HOST: str = "http://localhost:8000"
    PROJECT_NAME: str = "vietnamese-law-api"
    GEMINI_API_KEY: str
    MODEL: str = "gemini-2.5-flash-lite"
    QDRANT_PORT: int = 6333
    DOCS_ROOT: str = "./law_crawler/vbpl_documents"
    COLLECTION_NAME: str = "laws"
    EMBEDDING_MODEL_NAME: str = "Savoxism/vietnamese-legal-embedding-finetuned"
    EMBEDDINGS_FILE: str = "data/embeddings.pkl"
    DOCS_FILE: str = "data/documents.json"

    # JWT Settings
    JWT_SECRET_KEY: str = "change-me-in-production-use-env-var"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # Database Settings
    DATABASE_URL: str = "sqlite:///./law.db"


config = Config()