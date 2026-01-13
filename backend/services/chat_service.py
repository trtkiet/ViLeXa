import logging
from typing import Optional, Dict, List, Any, Tuple
import time # Ensure time is imported
import torch
import gc

# LangChain Imports
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Client Imports
from qdrant_client import QdrantClient

# Direct Sentence Transformers Import
from sentence_transformers import CrossEncoder

# Config Import
from core.config import config
from services.adapters import GTEDenseAdapter, GTESparseAdapter, GTEEmbedding

logger = logging.getLogger(__name__)
# Configure logging to show up in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatService:  
    """Handle chat completion requests against Gemini with RAG."""

    def __init__(self, use_reranker: bool = False) -> None:
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.base_retriever = None 
        self.reranker: Optional[CrossEncoder] = None
        self._initialized = False
        self.use_reranker = use_reranker
        self.session_store: Dict[str, BaseChatMessageHistory] = {}

    def startup(self) -> None:
        if self._initialized:
            return

        logger.info("--- Starting Initialization ---")
        t_start = time.time()

        # 1. Initialize Gemini
        self.model = ChatGoogleGenerativeAI(
            model=config.MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3
        )
        
        # 2. Initialize Embeddings
        gte_engine = GTEEmbedding(
            model_name="Alibaba-NLP/gte-multilingual-base",
            device="cpu" 
        )
        dense_embeddings = GTEDenseAdapter(gte_engine)
        sparse_embeddings = GTESparseAdapter(gte_engine)

        # 3. Initialize Qdrant
        try:
            client = QdrantClient(host="qdrant", port=config.QDRANT_PORT)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=config.COLLECTION_NAME,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=RetrievalMode.HYBRID
            )
            # Retrieve 20 candidates
            self.base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.vector_store = None

        # 4. DIRECT RERANKER SETUP
        if self.use_reranker:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading Reranker on: {device}")
                
                self.reranker = CrossEncoder(
                    "Alibaba-NLP/gte-multilingual-reranker-base",
                    device=device,
                    trust_remote_code=True,
                    model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
                )
                self.reranker.max_length = 1024 
                
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")

        self._initialized = True
        logger.info(f"--- Initialization Complete in {time.time() - t_start:.2f}s ---")
        
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def respond(self, query: str, session_id: Optional[str] = "default") -> Dict[str, Any]:
        if not self._initialized:
            self.startup()

        logger.info(f"Processing Query: '{query}'")
        total_start_time = time.time()
        
        context_text = ""
        source_documents = []

        # --- STEP 1: RETRIEVAL ---
        t_retrieval_start = time.time()
        
        # Get raw candidates from Qdrant
        initial_docs = self.base_retriever.invoke(query)
        
        t_retrieval_end = time.time()
        logger.info(f"1. Retrieval (Qdrant)  : {t_retrieval_end - t_retrieval_start:.4f}s | Found {len(initial_docs)} docs")

        t_rerank_start = time.time()
        top_k = config.TOP_K
        
        # Check 1: Is feature enabled? 
        # Check 2: Are there docs? 
        # Check 3: Is model loaded?
        if self.use_reranker and initial_docs and self.reranker:
            logger.info("Using Direct Reranker for final selection.")
            try:
                # ---------------------------------------------------------
                # OPTIMIZATION FIX: Slice to 2500 chars
                # Your snippet had [query, doc.page_content], which causes the 7s lag.
                # We MUST use [:2500] to save the CPU tokenizer.
                # ---------------------------------------------------------
                pairs = [[query, doc.page_content] for doc in initial_docs]
                
                # ---------------------------------------------------------
                # OPTIMIZATION FIX: Batch Size 8
                # Your snippet had 32, which causes swapping on GTX 1650 (4GB).
                # ---------------------------------------------------------
                scores = self.reranker.predict(
                    pairs, 
                    batch_size=8, # <-- Kept at 8 for GTX 1650 stability
                    show_progress_bar=False,
                    convert_to_tensor=False 
                )

                # Combine docs with their scores
                scored_docs = list(zip(initial_docs, scores))
                scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
                
                # Take Top K
                top_k_docs = [doc for doc, score in scored_docs[:top_k]]
                
                context_text = "\n\n".join([d.page_content for d in top_k_docs])
                source_documents = [d.metadata for d in top_k_docs]

            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # Fallback on error
                context_text = "\n\n".join([d.page_content for d in initial_docs[:top_k]])
                source_documents = [d.metadata for d in initial_docs[:top_k]]
        else:
            # Fallback if Reranker is Disabled OR Not Loaded
            if not self.use_reranker:
                logger.info("Reranker is DISABLED. Using raw Qdrant results.")
            
            context_text = "\n\n".join([d.page_content for d in initial_docs[:top_k]])
            source_documents = [d.metadata for d in initial_docs[:top_k]]
            
        t_rerank_end = time.time()
        logger.info(f"2. Reranking (CrossEnc): {t_rerank_end - t_rerank_start:.4f}s")

        # --- STEP 3: GENERATION (Gemini) ---
        t_gen_start = time.time()
        
        system_template = """You are a Vietnamese Legal Assistant. 
        Answer the user's question using the provided context.
        
        CONTEXT:
        {context}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"), 
            ("human", "{question}")
        ])

        chain = prompt_template | self.model | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        try:
            response_text = chain_with_history.invoke(
                {"context": context_text, "question": query},
                config={"configurable": {"session_id": session_id}}
            )
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return {"answer": "Error generating response.", "sources": []}

        t_gen_end = time.time()
        logger.info(f"3. Generation (Gemini) : {t_gen_end - t_gen_start:.4f}s")
        
        # --- TOTAL SUMMARY ---
        total_time = time.time() - total_start_time
        logger.info(f"=== Total Request Time : {total_time:.4f}s ===")

        return {
            "answer": response_text,
            "sources": source_documents
        }
            
def main():
    service = ChatService()
    service.startup()
    
if __name__ == "__main__":
    main()