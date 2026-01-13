import logging
from typing import Optional, Dict, Any

# LangChain Imports
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Client Imports
from qdrant_client import QdrantClient

# Config Import
from core.config import config

logger = logging.getLogger(__name__)

class ChatService:
    """Handle chat completion requests against Gemini with RAG."""

    def __init__(self) -> None:
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self._initialized = False

        # --- MEMORY SETUP ---
        # For production, replace this dict with Redis or a Database
        self.session_store: Dict[str, BaseChatMessageHistory] = {}

    def startup(self) -> None:
        """Initialize the Gemini client, Qdrant, and Embedding model."""
        if self._initialized:
            return

        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        
        # 1. Initialize Gemini
        # NOTE: Parameter is 'google_api_key', not 'gemini_api_key'
        self.model = ChatGoogleGenerativeAI(
            model=config.MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3 # Low temperature for factual RAG responses
        )
        
        # 2. Initialize Qdrant & Embeddings
        try:
            # Try Docker hostname first, fall back to localhost
            try:
                client = QdrantClient(host="qdrant", port=config.QDRANT_PORT)
                client.get_collections()  # Test connection
                logger.info("Connected to Qdrant at 'qdrant' hostname")
            except Exception:
                client = QdrantClient(host="localhost", port=config.QDRANT_PORT)
                logger.info("Connected to Qdrant at 'localhost'")

            # We use LangChain's wrapper for the store
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=config.COLLECTION_NAME,
                embedding=HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
            )
            logger.info("Connected to Qdrant successfully.")
        except Exception as e:
            logger.error(f"Could not connect to Qdrant: {e}")
            # We might want to allow startup to finish without vector store
            # so the chat works (just without context)
            self.vector_store = None

        self._initialized = True
        logger.info("ChatService initialized")
        
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Internal helper to retrieve chat history for a specific session."""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def respond(self, query: str, session_id: Optional[str] = "default") -> Dict[str, Any]:
        """
        1. Retrieve relevant documents from Qdrant.
        2. Inject them into the prompt.
        3. Get answer from Gemini.
        """
        if not self._initialized:
            self.startup()

        # 1. Retrieval Step
        context_text = ""
        source_documents = []

        if self.vector_store:
            try:
                # Search for top 4 relevant chunks
                docs = self.vector_store.similarity_search(query, k=4)

                # Format context for the LLM
                context_text = "\n\n".join([d.page_content for d in docs])

                # Transform metadata to match frontend LawSource interface
                for doc in docs:
                    meta = doc.metadata
                    article_full = meta.get('article', '')
                    # Extract article number (e.g., "Điều 141" from "Điều 141: Tội hiếp dâm")
                    article_parts = article_full.split(':', 1)
                    article_num = article_parts[0].strip() if article_parts else ''

                    source_documents.append({
                        'chapter': meta.get('chapter', ''),
                        'section': meta.get('section', '') or '',
                        'article': article_num,
                        'article_title': article_full,
                        'clause': ', '.join(meta.get('included_clauses', [])),
                        'source_text': doc.page_content
                    })
                logger.info(f"[DEBUG] Retrieved {len(source_documents)} sources")
            except Exception as e:
                logger.error(f"Error retrieving from Qdrant: {e}")
                context_text = "No context available due to database error."

        # ===== DEBUG: Log what we send to Gemini =====
        logger.info("=" * 60)
        logger.info(f"[GEMINI INPUT] Session: {session_id}")
        logger.info(f"[GEMINI INPUT] Query: {query}")
        logger.info(f"[GEMINI INPUT] Context length: {len(context_text)} chars")
        if context_text:
            logger.info(f"[GEMINI INPUT] Context preview:\n{context_text[:1000]}...")
        else:
            logger.info("[GEMINI INPUT] Context: EMPTY (No RAG)")

        # Log history
        history = self._get_session_history(session_id)
        logger.info(f"[GEMINI INPUT] History messages: {len(history.messages)}")
        for i, msg in enumerate(history.messages):
            logger.info(f"[GEMINI INPUT] History[{i}] ({msg.type}): {str(msg.content)[:200]}...")
        logger.info("=" * 60)
        # ===== END DEBUG =====

        # 2. Prompt Construction (Now with History!)
        system_template = """You are a Vietnamese Legal Assistant. 
        Answer the user's question using the provided context.
        If the answer is not in the context, say so.
        
        CONTEXT:
        {context}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"), # <--- Memory injection point
            ("human", "{question}")
        ])

        # 3. Create the Chain
        chain = prompt_template | self.model | StrOutputParser()

        # 4. Wrap Chain with Memory Management
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history, # Logic to get/create history
            input_messages_key="question",
            history_messages_key="history",
        )

        try:
            # 5. Invoke with session_id config
            response_text = chain_with_history.invoke(
                {"context": context_text, "question": query},
                config={"configurable": {"session_id": session_id}} # <--- Critical for memory
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "Sorry, an error occurred.",
                "sources": []
            }

        return {
            "answer": response_text,
            "sources": source_documents
        }
        
            
def main():
    service = ChatService()
    service.startup()
    
if __name__ == "__main__":
    main()

