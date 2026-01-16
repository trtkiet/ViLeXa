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
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    DOCS_ROOT: str = "./law_crawler/vbpl_documents"
    COLLECTION_NAME: str = "laws"
    EMBEDDING_MODEL_NAME: str = "Alibaba-NLP/gte-multilingual-base"
    EMBEDDINGS_FILE: str = "data/embeddings.pkl"
    DOCS_FILE: str = "data/documents.json"

    # =========================================================================
    # Retrieval & Reranking Performance Settings
    # =========================================================================
    # These settings control the retrieval pipeline performance vs quality.
    #
    # RETRIEVAL_MODE: Retrieval strategy
    #   - "hybrid": Dense + Sparse vectors (best quality, default)
    #   - "dense": Dense vectors only (semantic search)
    #   - "sparse": Sparse vectors only (keyword/BM25-style matching)
    #
    # RETRIEVAL_K: Number of candidates retrieved from Qdrant and passed to
    #              the reranker. Higher values improve recall but slow down
    #              reranking significantly.
    #
    # TOP_K: Final number of documents returned to the LLM for generation.
    #        Should be <= RETRIEVAL_K.
    #
    # Performance benchmarks (GTX 1650, 4GB VRAM, full document content):
    #   RETRIEVAL_K=5:  ~0.9s per query
    #   RETRIEVAL_K=10: ~1.8s per query
    #   RETRIEVAL_K=20: ~3.7s per query
    #
    # Without reranker, RETRIEVAL_K only affects initial recall (fast).
    #
    # Recommendations:
    #   - Production (balanced): RETRIEVAL_K=10, TOP_K=4
    #   - Benchmarks (quality):  RETRIEVAL_K=20, TOP_K=5
    # =========================================================================
    RETRIEVAL_MODE: str = "hybrid"  # "hybrid", "dense", or "sparse"
    RETRIEVAL_K: int = 10  # Candidates to retrieve from Qdrant and rerank
    TOP_K: int = 3  # Final docs sent to LLM

    # JWT Settings
    JWT_SECRET_KEY: str = "change-me-in-production-use-env-var"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # Database Settings
    DATABASE_URL: str = "sqlite:///./law.db"

    # =========================================================================
    # Vietnamese Embedding Settings (BGE-M3 based)
    # =========================================================================
    # Settings for the Vietnamese Embedding pipeline using
    # AITeamVN/Vietnamese_Embedding_v2 (fine-tuned BGE-M3)
    #
    # VIETNAMESE_EMBEDDING_MODEL: HuggingFace model name
    # VIETNAMESE_COLLECTION_NAME: Qdrant collection name for Vietnamese embeddings
    # VIETNAMESE_DENSE_VECTOR_SIZE: Dimension of dense vectors (1024 for BGE-M3)
    # =========================================================================
    VIETNAMESE_EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding_v2"
    VIETNAMESE_COLLECTION_NAME: str = "laws"
    VIETNAMESE_DENSE_VECTOR_SIZE: int = 1024

    # =========================================================================
    # BGE-M3 Embedding Settings (BAAI/bge-m3)
    # =========================================================================
    # Settings for the BGE-M3 pipeline using the original BAAI/bge-m3 model.
    # Multilingual model supporting 100+ languages with hybrid retrieval.
    #
    # BGEM3_EMBEDDING_MODEL: HuggingFace model name
    # BGEM3_COLLECTION_NAME: Qdrant collection name for BGE-M3 embeddings
    # BGEM3_DENSE_VECTOR_SIZE: Dimension of dense vectors (1024 for BGE-M3)
    # =========================================================================
    BGEM3_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    BGEM3_COLLECTION_NAME: str = "laws_bge_m3"
    BGEM3_DENSE_VECTOR_SIZE: int = 1024


config = Config()
