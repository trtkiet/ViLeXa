"""Hybrid RAG Pipeline using Qdrant vector store with optional reranking."""

import logging
import time
from typing import Dict, Any, Optional, List

import torch

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from core.config import config
from services.adapters import GTEDenseAdapter, GTESparseAdapter, GTEEmbedding
from services.pipelines.base import RAGPipeline

logger = logging.getLogger(__name__)


class HybridRAGPipeline(RAGPipeline):
    """
    Hybrid RAG pipeline using Qdrant with dense + sparse vectors.

    Features:
    - Hybrid retrieval (dense + sparse embeddings via GTE)
    - Optional cross-encoder reranking
    - Gemini LLM for generation
    """

    def __init__(self, use_reranker: bool = False) -> None:
        """
        Initialize the pipeline.

        Args:
            use_reranker: Whether to enable cross-encoder reranking.
        """
        self.vector_store: Optional[QdrantVectorStore] = None
        self.model: Optional[ChatGoogleGenerativeAI] = None
        self.base_retriever = None
        self.reranker: Optional[CrossEncoder] = None
        self._initialized = False
        self.use_reranker = use_reranker

    @property
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        return self._initialized

    def startup(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("--- Starting HybridRAGPipeline Initialization ---")
        t_start = time.time()

        # 1. Initialize Gemini LLM
        self.model = ChatGoogleGenerativeAI(
            model=config.MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3,
        )

        # 2. Initialize Embeddings
        gte_engine = GTEEmbedding(
            model_name="Alibaba-NLP/gte-multilingual-base",
            device="cpu",
        )
        dense_embeddings = GTEDenseAdapter(gte_engine)
        sparse_embeddings = GTESparseAdapter(gte_engine)

        # 3. Initialize Qdrant Vector Store
        try:
            client = QdrantClient(host="qdrant", port=config.QDRANT_PORT)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=config.COLLECTION_NAME,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                vector_name="dense",
                sparse_vector_name="sparse",
                retrieval_mode=RetrievalMode.HYBRID,
            )
            # Retrieve 20 candidates for reranking
            self.base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 20}
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.vector_store = None

        # 4. Initialize Reranker (if enabled)
        if self.use_reranker:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading Reranker on: {device}")

                self.reranker = CrossEncoder(
                    "Alibaba-NLP/gte-multilingual-reranker-base",
                    device=device,
                    trust_remote_code=True,
                    model_kwargs=(
                        {"torch_dtype": torch.float16} if device == "cuda" else {}
                    ),
                )
                self.reranker.max_length = 1024
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")

        self._initialized = True
        logger.info(
            f"--- HybridRAGPipeline Initialization Complete in {time.time() - t_start:.2f}s ---"
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
        top_k = config.TOP_K

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

        if self.use_reranker and initial_docs and self.reranker:
            logger.info("Using Direct Reranker for final selection.")
            try:
                pairs = [[query, doc.page_content] for doc in initial_docs]

                scores = self.reranker.predict(
                    pairs,
                    batch_size=8,  # Optimized for GTX 1650 (4GB)
                    show_progress_bar=False,
                    convert_to_tensor=False,
                )

                # Combine docs with scores and sort
                scored_docs = list(zip(initial_docs, scores))
                scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

                # Take Top K
                top_k_docs = [doc for doc, score in scored_docs[:top_k]]

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
            if not self.use_reranker:
                logger.info("Reranker is DISABLED. Using raw Qdrant results.")

            context_text = "\n\n".join([d.page_content for d in initial_docs[:top_k]])
            source_documents = [d.metadata for d in initial_docs[:top_k]]

        t_rerank_end = time.time()
        logger.info(f"2. Reranking (CrossEnc): {t_rerank_end - t_rerank_start:.4f}s")

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

    def run(
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
