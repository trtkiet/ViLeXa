from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Config(BaseSettings):
    APP_NAME: str = "Vietnamese Law API"
    API_STR: str = "/api/v1"
    SERVER_HOST: str = "http://localhost:8000"
    PROJECT_NAME: str = "vietnamese-law-api"
    GEMINI_API_KEY: str
    MODEL: str = "gemini-2.5-flash"
    QDRANT_PORT: int = 6333
    DOCS_ROOT: str = "./law_crawler/vbpl_documents"

    
config = Config()