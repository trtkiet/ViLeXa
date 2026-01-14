"""Chat service that orchestrates RAG pipelines for conversational AI."""

import logging
from typing import Optional, Dict, Any

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from services.pipelines import RAGPipeline, HybridRAGPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChatService:
    """
    Chat service that manages conversation sessions and delegates
    to a RAGPipeline for retrieval and generation.

    This service handles:
    - Session/conversation history management
    - Pipeline lifecycle (startup/shutdown)
    - Routing queries through the configured pipeline

    The actual RAG logic (retrieval, reranking, generation) is handled
    by the injected RAGPipeline implementation, allowing different
    strategies to be swapped easily.
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None) -> None:
        """
        Initialize the chat service.

        Args:
            pipeline: The RAG pipeline to use for processing queries.
                     Defaults to HybridRAGPipeline if not provided.
        """
        # Default to HybridRAGPipeline for backward compatibility
        self.pipeline = pipeline or HybridRAGPipeline(use_reranker=False)
        self.session_store: Dict[str, BaseChatMessageHistory] = {}

    def startup(self) -> None:
        """Initialize the underlying pipeline."""
        self.pipeline.startup()

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create chat history for a session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            The chat message history for this session.
        """
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def respond(
        self, query: str, session_id: Optional[str] = "default"
    ) -> Dict[str, Any]:
        """
        Process a user query and return a response.

        Args:
            query: The user's question or message.
            session_id: Identifier for the conversation session.
                       Defaults to "default".

        Returns:
            Dictionary containing:
                - "answer": The generated response text.
                - "sources": List of source document metadata.
        """
        # Get history for this session
        history = self._get_session_history(session_id)

        # Run the pipeline with history
        result = self.pipeline.run(query, history=history.messages)

        # Update history with this exchange
        # Note: The API layer (chat.py) also manages history persistence to DB,
        # but we update the in-memory store here for consistency
        history.add_user_message(query)
        history.add_ai_message(result["answer"])

        return result


def main():
    """Test the chat service."""
    service = ChatService()
    service.startup()


if __name__ == "__main__":
    main()
