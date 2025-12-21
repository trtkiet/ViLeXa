import logging
from typing import Optional

from google import genai

from backend.core.config import config

logger = logging.getLogger(__name__)


class ChatService:
    """Handle chat completion requests against Gemini."""

    def __init__(self) -> None:
        self.client: Optional[genai.Client] = None
        self._initialized = False

    def startup(self) -> None:
        """Initialize the Gemini client once for the process."""
        if self._initialized:
            return

        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._initialized = True
        logger.info("ChatService initialized")

    def respond(self, query: str) -> str:
        # if not self._initialized or self.client is None:
        #     # Ensure the service is ready even if startup hook was missed (e.g., in tests).
        #     self.startup()

        response = self.client.models.generate_content(
            model=config.MODEL,
            contents=query,
        )
        return response.text

