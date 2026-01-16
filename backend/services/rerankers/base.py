"""Abstract base class for document rerankers."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseReranker(ABC):
    """
    Abstract base class for document rerankers.

    Rerankers take a query and a list of retrieved documents, then reorder
    the documents by relevance to the query. This is typically more accurate
    than initial retrieval scoring but slower.

    Implementations can use:
    - Cross-encoder models (e.g., sentence-transformers CrossEncoder)
    - API-based rerankers (e.g., Cohere, Jina)
    - LLM-based rerankers
    """

    @abstractmethod
    def startup(self) -> None:
        """
        Initialize the reranker.

        This includes:
        - Loading model weights
        - Setting up device (CPU/GPU)
        - Warmup inference (if applicable)
        """
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's query/question.
            documents: List of documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            List of top_k documents sorted by relevance (highest first).
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the reranker has been initialized and is ready to use."""
        pass
