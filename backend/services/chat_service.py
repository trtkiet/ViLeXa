"""Chat service that orchestrates RAG pipelines for conversational AI."""

import logging
from typing import Optional, Dict, Any, Iterator

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from services.pipelines import RAGPipeline, GTEPipeline

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
                     Defaults to GTEPipeline if not provided.
        """
        # Default to GTEPipeline
        self.pipeline = pipeline or GTEPipeline()
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
        result = self.pipeline.respond(query, history=history.messages)

        # Update history with this exchange
        # Note: The API layer (chat.py) also manages history persistence to DB,
        # but we update the in-memory store here for consistency
        history.add_user_message(query)
        history.add_ai_message(result["answer"])

        return result

    def stream_respond(
        self, query: str, session_id: Optional[str] = "default"
    ) -> Iterator[Dict[str, Any]]:
        """
        Process a user query and stream the response.

        Args:
            query: The user's question or message.
            session_id: Identifier for the conversation session.

        Yields:
            Dictionaries with streaming data (sources, tokens, done/error).
        """
        # Get history for this session
        history = self._get_session_history(session_id or "default")

        full_answer = ""

        # Stream from pipeline
        for chunk in self.pipeline.stream_respond(query, history=history.messages):
            if chunk.get("type") == "done":
                full_answer = chunk.get("answer", "")
            yield chunk

        # Update history with this exchange after streaming completes
        if full_answer:
            history.add_user_message(query)
            history.add_ai_message(full_answer)


def main():
    """Test the chat service."""
    service = ChatService()
    service.startup()


if __name__ == "__main__":
    main()
