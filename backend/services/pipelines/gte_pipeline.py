"""GTE-based RAG Pipeline using Qdrant vector store with configurable retrieval modes."""

import logging
import time
from typing import Dict, Any, Optional, List, Iterator

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from qdrant_client import QdrantClient

from core.config import config
from services.adapters import GTEDenseAdapter, GTESparseAdapter, GTEEmbedding
from services.pipelines.base import RAGPipeline
from services.rerankers import BaseReranker, CrossEncoderReranker

logger = logging.getLogger(__name__)

# Map string mode to langchain_qdrant RetrievalMode enum
_RETRIEVAL_MODE_MAP = {
    "hybrid": RetrievalMode.HYBRID,
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
}


class GTEPipeline(RAGPipeline):
    """
    RAG pipeline using GTE embeddings with Qdrant vector store.

    Features:
    - Configurable retrieval mode: hybrid, dense, or sparse
    - Optional cross-encoder reranking
    - Gemini LLM for generation

    Args:
        retrieval_mode: "hybrid" (dense+sparse), "dense", or "sparse".
                       Defaults to config.RETRIEVAL_MODE.
        use_reranker: Enable cross-encoder reranking (creates default reranker).
        reranker: Optional custom reranker instance (takes precedence over use_reranker).
        collection_name: Qdrant collection (defaults to config.COLLECTION_NAME).
        qdrant_host: Qdrant host (defaults to "qdrant" for Docker).
        qdrant_port: Qdrant port (defaults to config.QDRANT_PORT).
        skip_llm: Skip LLM initialization for retrieval-only benchmarks.
        retrieval_k: Number of candidates to retrieve (defaults to config.RETRIEVAL_K).
        top_k: Number of documents after reranking (defaults to config.TOP_K).
    """

    def __init__(
        self,
        retrieval_mode: Optional[str] = None,
        use_reranker: bool = False,
        reranker: Optional[BaseReranker] = None,
        collection_name: Optional[str] = None,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        skip_llm: bool = False,
        retrieval_k: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            retrieval_mode: Retrieval strategy - "hybrid", "dense", or "sparse".
            use_reranker: Whether to enable cross-encoder reranking (creates default).
            reranker: Optional custom reranker instance (takes precedence over use_reranker).
            collection_name: Qdrant collection name (defaults to config.COLLECTION_NAME).
            qdrant_host: Qdrant host (defaults to "qdrant" for Docker).
            qdrant_port: Qdrant port (defaults to config.QDRANT_PORT).
            skip_llm: Skip LLM initialization (for retrieval-only benchmarks).
            retrieval_k: Number of candidates to retrieve (defaults to config.RETRIEVAL_K).
            top_k: Number of documents after reranking (defaults to config.TOP_K).
        """
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.base_retriever = None
        self.reranker: Optional[BaseReranker] = reranker
        self._initialized = False
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker or (reranker is not None)
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.skip_llm = skip_llm
        self._retrieval_k = retrieval_k
        self._top_k = top_k

    @property
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        return self._initialized

    def startup(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("--- Starting GTEPipeline Initialization ---")
        t_start = time.time()

        # Resolve configuration with defaults
        retrieval_mode = self.retrieval_mode or config.RETRIEVAL_MODE
        collection_name = self.collection_name or config.COLLECTION_NAME
        qdrant_host = self.qdrant_host or config.QDRANT_HOST
        qdrant_port = self.qdrant_port or config.QDRANT_PORT
        self._resolved_retrieval_k = self._retrieval_k or config.RETRIEVAL_K
        self._resolved_top_k = self._top_k or config.TOP_K

        # Validate retrieval mode
        if retrieval_mode not in _RETRIEVAL_MODE_MAP:
            raise ValueError(
                f"Invalid retrieval_mode: {retrieval_mode}. "
                f"Must be one of: {list(_RETRIEVAL_MODE_MAP.keys())}"
            )

        # 1. Initialize Gemini LLM (skip if retrieval-only mode)
        if not self.skip_llm:
            self.model = ChatGoogleGenerativeAI(
                model=config.MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0.3,
            )
        else:
            logger.info("Skipping LLM initialization (retrieval-only mode)")

        # 2. Initialize GTE Embeddings (only load what's needed for the mode)
        gte_engine = GTEEmbedding(
            model_name="Alibaba-NLP/gte-multilingual-base",
            device="cpu",
        )

        dense_embeddings = None
        sparse_embeddings = None

        if retrieval_mode in ("hybrid", "dense"):
            dense_embeddings = GTEDenseAdapter(gte_engine)
            logger.info("Dense embeddings enabled")

        if retrieval_mode in ("hybrid", "sparse"):
            sparse_embeddings = GTESparseAdapter(gte_engine)
            logger.info("Sparse embeddings enabled")

        # 3. Initialize Qdrant Vector Store
        try:
            logger.info(f"Trying to connect to: {qdrant_host}:{qdrant_port}")
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=_RETRIEVAL_MODE_MAP[retrieval_mode],
            )
            # Retrieve RETRIEVAL_K candidates for reranking
            self.base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self._resolved_retrieval_k}
            )
            logger.info(
                f"Connected to Qdrant: {qdrant_host}:{qdrant_port}/{collection_name} "
                f"(mode={retrieval_mode}, retrieval_k={self._resolved_retrieval_k}, top_k={self._resolved_top_k})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.vector_store = None

        # 4. Initialize Reranker (if enabled and not already provided)
        if self.use_reranker and self.reranker is None:
            self.reranker = CrossEncoderReranker()

        if self.reranker is not None and not self.reranker.is_initialized:
            try:
                self.reranker.startup()
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")
                self.reranker = None

        self._initialized = True
        logger.info(
            f"--- GTEPipeline Initialization Complete in {time.time() - t_start:.2f}s ---"
        )

    def _retrieve_and_rerank(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve documents and optionally rerank them.

        Args:
            query: The user's query.

        Returns:
            Tuple of (context_text, source_documents).
        """
        context_text = ""
        source_documents: List[Dict[str, Any]] = []
        top_k = self._resolved_top_k

        # Step 1: Retrieval from Qdrant
        t_retrieval_start = time.time()
        initial_docs = self.base_retriever.invoke(query)
        t_retrieval_end = time.time()
        logger.info(
            f"1. Retrieval (Qdrant)  : {t_retrieval_end - t_retrieval_start:.4f}s | "
            f"Found {len(initial_docs)} docs"
        )

        # Step 2: Reranking (if enabled)
        t_rerank_start = time.time()

        if self.reranker and self.reranker.is_initialized and initial_docs:
            logger.info("Using Reranker for final selection.")
            try:
                top_k_docs = self.reranker.rerank(query, initial_docs, top_k)
                context_text = "\n\n".join([d.page_content for d in top_k_docs])
                source_documents = [d.metadata for d in top_k_docs]
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # Fallback on error
                context_text = "\n\n".join(
                    [d.page_content for d in initial_docs[:top_k]]
                )
                source_documents = [d.metadata for d in initial_docs[:top_k]]
        else:
            # Fallback if Reranker is disabled or not loaded
            if not self.reranker:
                logger.info("Reranker is DISABLED. Using raw Qdrant results.")

            context_text = "\n\n".join([d.page_content for d in initial_docs[:top_k]])
            source_documents = [d.metadata for d in initial_docs[:top_k]]

        t_rerank_end = time.time()
        logger.info(f"2. Reranking           : {t_rerank_end - t_rerank_start:.4f}s")

        return context_text, source_documents

    def _generate(
        self,
        query: str,
        context_text: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> str:
        """
        Generate response using Gemini LLM.

        Args:
            query: The user's question.
            context_text: Retrieved context for the LLM.
            history: Optional chat history.

        Returns:
            The generated response text.
        """
        if self.model is None:
            raise RuntimeError(
                "LLM not initialized. Set skip_llm=False to enable generation."
            )

        t_gen_start = time.time()

        system_template = """You are a Vietnamese Legal Assistant. 
        Answer the user's question using the provided context.
        
        CONTEXT:
        {context}"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt_template | self.model | StrOutputParser()

        try:
            # Build messages for history placeholder
            history_messages = history if history else []

            response_text = chain.invoke(
                {
                    "context": context_text,
                    "question": query,
                    "history": history_messages,
                }
            )
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return "Error generating response."

        t_gen_end = time.time()
        logger.info(f"3. Generation (Gemini) : {t_gen_end - t_gen_start:.4f}s")

        return response_text

    def retrieve_context(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Retrieve context documents without generating a response.

        Args:
            query: The user's question.

        Returns:
            Dictionary with "context" and "sources" keys.
        """
        if not self._initialized:
            self.startup()

        logger.info(f"Retrieving context for: '{query}'")

        context_text, source_documents = self._retrieve_and_rerank(query)

        return {
            "context": context_text,
            "sources": source_documents,
        }

    def respond(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.

        Args:
            query: The user's question.
            history: Optional list of previous chat messages.

        Returns:
            Dictionary with "answer" and "sources" keys.
        """
        if not self._initialized:
            self.startup()

        logger.info(f"Processing Query: '{query}'")
        total_start_time = time.time()

        # Retrieve and rerank
        context_text, source_documents = self._retrieve_and_rerank(query)

        # Generate response
        response_text = self._generate(query, context_text, history)

        # Log total time
        total_time = time.time() - total_start_time
        logger.info(f"=== Total Request Time : {total_time:.4f}s ===")

        return {
            "answer": response_text,
            "sources": source_documents,
        }

    def stream_respond(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Execute the RAG pipeline with streaming response.

        Args:
            query: The user's question.
            history: Optional list of previous chat messages.

        Yields:
            Dictionaries with streaming data.
        """
        if not self._initialized:
            self.startup()

        if self.model is None:
            yield {"type": "error", "error": "LLM not initialized (skip_llm=True)"}
            return

        logger.info(f"[Stream] Processing Query: '{query}'")
        total_start_time = time.time()

        # Retrieve and rerank first
        context_text, source_documents = self._retrieve_and_rerank(query)

        # Yield sources immediately
        yield {"type": "sources", "sources": source_documents}

        # Stream generation
        t_gen_start = time.time()

        system_template = """Bạn là một trợ lý AI về pháp luật Việt Nam, hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không đủ để trả lời, hãy nói rằng bạn không có đủ thông tin, không tự bịa thêm.
Hãy trích dẫn cụ thể điều luật, khoản, điểm khi trả lời.

Ngữ cảnh:
{context}"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt_template | self.model

        try:
            history_messages = history if history else []
            full_response = ""

            for chunk in chain.stream(
                {
                    "context": context_text,
                    "question": query,
                    "history": history_messages,
                }
            ):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_response += token
                    yield {"type": "token", "token": token}

            t_gen_end = time.time()
            logger.info(f"[Stream] Generation time: {t_gen_end - t_gen_start:.4f}s")

            # Yield final complete answer
            yield {"type": "done", "answer": full_response}

        except Exception as e:
            logger.error(f"[Stream] Gemini Error: {e}")
            yield {"type": "error", "error": str(e)}

        total_time = time.time() - total_start_time
        logger.info(f"[Stream] Total Request Time: {total_time:.4f}s")
