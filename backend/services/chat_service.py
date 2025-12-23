import logging
from typing import Optional, Dict, List, Any

from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.core.config import config

logger = logging.getLogger(__name__)


class ChatService:
    """Handle chat completion requests against Gemini with RAG."""

    def __init__(self) -> None:
        self.client: Optional[genai.Client] = None
        self.qdrant: Optional[QdrantClient] = None
        self.embedder: Optional[SentenceTransformer] = None
        self._initialized = False

    def startup(self) -> None:
        """Initialize the Gemini client, Qdrant, and Embedding model."""
        if self._initialized:
            return

        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        
        # Initialize Qdrant
        # Use 'qdrant' hostname if running in Docker, or 'localhost' if local
        # We'll assume Docker for now based on the setup
        try:
            self.qdrant = QdrantClient(host="qdrant", port=6333)
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")

        # Initialize Embedding Model
        try:
            self.embedder = SentenceTransformer('minhquan6203/paraphrase-vietnamese-law')
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")

        self._initialized = True
        logger.info("ChatService initialized")

    def respond(self, query: str) -> Dict[str, Any]:
        if not self._initialized:
            self.startup()

        context_str = ""
        sources = []

        # RAG Step: Retrieve context if Qdrant and Embedder are ready
        if self.qdrant and self.embedder:
            try:
                query_vector = self.embedder.encode(query).tolist()
                search_results = self.qdrant.search(
                    collection_name="laws",
                    query_vector=query_vector,
                    limit=5
                )
                
                # Deduplicate articles to avoid repeating the same parent context
                seen_articles = set()
                context_parts = []
                
                for res in search_results:
                    payload = res.payload
                    article_id = f"{payload.get('law_id')}_{payload.get('article')}"
                    
                    # Add to sources list
                    sources.append(payload)
                    
                    # Add to context (Parent Article Text)
                    # We use the full article text if available, or just the clause text
                    text_to_add = payload.get('source_text', payload.get('text', ''))
                    
                    if article_id not in seen_articles:
                        context_parts.append(text_to_add)
                        seen_articles.add(article_id)
                
                context_str = "\n\n".join(context_parts)
                
            except Exception as e:
                logger.error(f"RAG Retrieval failed: {e}")

        # Construct System Prompt
        system_prompt = f"""
You are a Vietnamese Legal Assistant. Use the following context to answer the user's question.
If the answer is not in the context, state that you cannot find it in the provided documents.
Cite the Article Number (e.g., [Điều 5 - Luật Lao động]) for every claim you make.

CONTEXT:
{context_str}
"""

        response = self.client.models.generate_content(
            model=config.MODEL,
            contents=[system_prompt, query],
        )
        
        return {
            "reply": response.text,
            "sources": sources
        }

