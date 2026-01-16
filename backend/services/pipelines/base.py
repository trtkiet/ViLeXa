"""Abstract base class for RAG pipelines."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator

from langchain_core.messages import BaseMessage


class RAGPipeline(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) pipelines.

    Implementations should handle the full flow:
    - Document retrieval from vector store
    - Optional reranking
    - LLM generation with context

    This allows swapping different RAG strategies while maintaining
    a consistent interface for the ChatService.
    """

    @abstractmethod
    def startup(self) -> None:
        """
        Initialize all pipeline components.

        This includes:
        - Embedding models
        - Vector store connections
        - LLM initialization
        - Reranker models (if applicable)
        """
        pass

    @abstractmethod
    def respond(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.

        Args:
            query: The user's question/input.
            history: Optional list of previous chat messages for context.

        Returns:
            A dictionary containing:
                - "answer": The generated response text.
                - "sources": List of source document metadata.
        """
        pass

    @abstractmethod
    def retrieve_context(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Retrieve context documents without generating a response.

        Args:
            query: The user's question/input.

        Returns:
            A dictionary containing:
                - "context": The retrieved context text.
                - "sources": List of source document metadata.
        """
        pass

    @abstractmethod
    def stream_respond(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Execute the RAG pipeline with streaming response.

        Args:
            query: The user's question/input.
            history: Optional list of previous chat messages for context.

        Yields:
            Dictionaries containing either:
                - {"type": "sources", "sources": [...]} - Source documents (first yield)
                - {"type": "token", "token": "..."} - Generated tokens
                - {"type": "done", "answer": "..."} - Final complete answer
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized and is ready to use."""
        pass
