"""Cross-encoder based reranker implementation."""

import logging
import time
from typing import List, Optional

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from services.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using sentence-transformers CrossEncoder.

    Uses a cross-encoder model to score query-document pairs and return
    the top_k most relevant documents.

    Args:
        model_name: HuggingFace model name for the cross-encoder.
        max_length: Maximum sequence length for the model.
        device: Device to run inference on ("cuda", "cpu", or None for auto-detect).
        batch_size: Batch size for prediction. None means all pairs in one batch.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length: int = 1024,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the CrossEncoderReranker.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum input sequence length.
            device: Device for inference (None for auto-detect).
            batch_size: Batch size for prediction (None for all-at-once).
        """
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the reranker has been initialized."""
        return self._initialized

    def startup(self) -> None:
        """
        Initialize the CrossEncoder model.

        Includes:
        - CUDA availability detection with logging
        - Model loading with appropriate dtype
        - GPU warmup for CUDA kernel compilation
        """
        if self._initialized:
            return

        logger.info(f"Loading CrossEncoderReranker: {self.model_name}")
        t_start = time.time()

        # Device detection
        if self._device is not None:
            device = self._device
            cuda_available = device == "cuda"
        else:
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"

        # Log device info
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(
                    f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB) - using GPU"
                )
            except Exception:
                logger.info("CUDA available - using GPU")
        else:
            logger.warning(
                "CUDA not available! Reranker will run on CPU (slower). "
                "Check NVIDIA Container Toolkit installation."
            )

        logger.info(f"Loading Reranker on: {device}")

        try:
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs=({"dtype": torch.float16} if cuda_available else {}),
            )
            self._model.max_length = self.max_length

            # Warmup inference to compile CUDA kernels
            if cuda_available:
                logger.info("Warming up reranker (first inference is slow)...")
                with torch.inference_mode():
                    _ = self._model.predict(
                        [["warmup query", "warmup document"]],
                        show_progress_bar=False,
                    )
                logger.info("Reranker warmup complete")

            self._initialized = True
            logger.info(f"CrossEncoderReranker loaded in {time.time() - t_start:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load CrossEncoderReranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's query.
            documents: List of documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            List of top_k documents sorted by relevance score (highest first).
        """
        if not self._initialized or self._model is None:
            raise RuntimeError(
                "CrossEncoderReranker not initialized. Call startup() first."
            )

        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents, returning top {top_k}")
        t_start = time.time()

        # Build query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Determine batch size
        batch_size = self.batch_size if self.batch_size else len(pairs)

        # Score all pairs
        t_predict_start = time.time()
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
        t_predict_end = time.time()
        logger.info(
            f"   Reranker predict(): {t_predict_end - t_predict_start:.4f}s "
            f"for {len(pairs)} pairs"
        )

        # Combine docs with scores and sort by score descending
        scored_docs = list(zip(documents, scores))
        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Take top_k
        top_k_docs = [doc for doc, score in scored_docs[:top_k]]

        logger.info(f"Reranking complete in {time.time() - t_start:.4f}s")

        return top_k_docs
