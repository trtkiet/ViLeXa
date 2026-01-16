import numpy as np
import torch
import gc
from typing import List, Dict, Tuple, Any
from collections import defaultdict

# LangChain / Qdrant Imports
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings
from qdrant_client.http import models as rest

# Transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
from transformers.utils import is_torch_npu_available


class GTEEmbedding(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        normalized: bool = True,
        use_fp16: bool = False,
        device: str = None,
    ):
        super().__init__()
        self.normalized = normalized

        # 1. Device Setup
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False  # CPU doesn't support fp16 well usually

        self.use_fp16 = use_fp16

        # 2. Model Loading
        # Note: We use AutoModelForTokenClassification if you are treating logits as SPLADE weights,
        # otherwise AutoModel is standard for Dense.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
        )
        self.model.eval()  # Explicitly set to eval mode
        self.model.to(self.device)

    def _process_token_weights(
        self, token_weights: np.ndarray, input_ids: list
    ) -> Dict[int, float]:
        """
        Aggregates weights for identical token IDs.
        Returns Dict[int, float] to prevent Duplicate Index errors in Qdrant.
        """
        result = defaultdict(float)
        unused_tokens = set(
            [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ]
        )

        for w, idx in zip(token_weights, input_ids):
            idx = int(idx)  # Ensure standard python int
            if idx not in unused_tokens and w > 0:
                # MAX Strategy: If token appears twice, keep max weight
                if w > result[idx]:
                    result[idx] = float(w)
                # SUM Strategy (Alternative): result[idx] += float(w)

        return dict(result)

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        dimension: int = None,
        max_length: int = 8192,
        batch_size: int = 16,
        return_dense: bool = True,
        return_sparse: bool = False,
    ) -> Dict[str, Any]:
        """
        Main encoding function. Wraps _encode with batching and memory cleanup.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_dense_vecs = []
        all_token_weights = []

        # Batch processing
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = self._encode_batch(
                batch, dimension, max_length, return_dense, return_sparse
            )

            if return_dense:
                all_dense_vecs.append(results["dense_embeddings"])
            if return_sparse:
                all_token_weights.extend(results["token_weights"])

            # MEMORY CLEANUP: Clear CUDA cache after every batch
            del results
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        final_output = {}
        if return_dense and all_dense_vecs:
            final_output["dense_embeddings"] = torch.cat(all_dense_vecs, dim=0)
        else:
            final_output["dense_embeddings"] = []

        if return_sparse:
            final_output["token_weights"] = all_token_weights
        else:
            final_output["token_weights"] = []

        return final_output

    @torch.no_grad()
    def _encode_batch(
        self,
        texts: List[str],
        dimension: int = None,
        max_length: int = 1024,
        return_dense: bool = True,
        return_sparse: bool = False,
    ):
        # Tokenize
        text_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        text_input = {k: v.to(self.device) for k, v in text_input.items()}

        # Forward Pass
        model_out = self.model(**text_input, return_dict=True)

        output = {}

        # DENSE LOGIC
        if return_dense:
            # Taking CLS token (index 0)
            dense_vecs = model_out.last_hidden_state[:, 0, :]
            if dimension:
                dense_vecs = dense_vecs[:, :dimension]
            if self.normalized:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            # Move to CPU immediately to free GPU memory
            output["dense_embeddings"] = dense_vecs.cpu()

        # SPARSE LOGIC (SPLADE style)
        if return_sparse:
            # ReLU on logits to get weights
            weights = torch.relu(model_out.logits).squeeze(-1)

            # Convert to numpy for processing
            weights_np = weights.detach().cpu().numpy()
            input_ids_np = text_input["input_ids"].cpu().numpy()

            # Map weights to IDs (Output is List[Dict[int, float]])
            output["token_weights"] = [
                self._process_token_weights(w, ids)
                for w, ids in zip(weights_np, input_ids_np)
            ]

        return output


# --- ADAPTERS ---


class GTEDenseAdapter(Embeddings):
    """Adapter for GTE Dense embeddings."""

    def __init__(self, gte_instance: GTEEmbedding):
        self.gte = gte_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.gte.encode(texts, return_dense=True, return_sparse=False)
        # Convert tensor to list
        return res["dense_embeddings"].numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        res = self.gte.encode([text], return_dense=True, return_sparse=False)
        return res["dense_embeddings"][0].numpy().tolist()


class GTESparseAdapter(SparseEmbeddings):
    """Adapter for GTE Sparse embeddings for Qdrant."""

    def __init__(self, gte_instance: GTEEmbedding):
        self.gte = gte_instance

    def _to_sparse_vector(
        self, token_weights_dict: Dict[int, float]
    ) -> rest.SparseVector:
        """
        Converts the Dict[int, float] from the engine into Qdrant SparseVector.
        Ensures indices are sorted (requirement for optimization).
        """
        # Sort by Index ID
        sorted_indices = sorted(token_weights_dict.keys())
        values = [token_weights_dict[idx] for idx in sorted_indices]

        return rest.SparseVector(indices=sorted_indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[rest.SparseVector]:
        res = self.gte.encode(texts, return_dense=False, return_sparse=True)
        return [self._to_sparse_vector(d) for d in res["token_weights"]]

    def embed_query(self, text: str) -> rest.SparseVector:
        res = self.gte.encode([text], return_dense=False, return_sparse=True)
        return self._to_sparse_vector(res["token_weights"][0])


# --- VIETNAMESE EMBEDDING (BGE-M3 based) ---


class VietnameseEmbedding:
    """
    Wrapper for AITeamVN/Vietnamese_Embedding_v2 (BGE-M3 fine-tuned).

    Uses FlagEmbedding.BGEM3FlagModel for dense + sparse embeddings.
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding_v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device

        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=device == "cuda",
            device=device,
        )
        self.model.model.eval()
        try:
            self.model.encode(["warmup"], batch_size=1, max_length=32)
        except Exception as e:
            print(f"Warmup failed: {e}")
        self.tokenizer = self.model.tokenizer

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        max_length: int = 8192,
        batch_size: int = 16,
    ) -> dict[str, any]:
        """
        Encode texts using BGE-M3 model.

        Returns:
            Dict with "dense_embeddings" and/or "sparse_embeddings" keys.
        """
        if isinstance(texts, str):
            texts = [texts]

        output = {}

        # Ensure eval mode and no_grad for determinism
        self.model.model.eval()
        with torch.no_grad():
            # Single encode call for efficiency - BGE-M3 returns dict with 'dense_vecs' and 'lexical_weights'
            results = self.model.encode(
                texts,
                batch_size=min(batch_size, len(texts)),
                max_length=max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"][0].tolist()
        print(f"Tokens Sample: {tokens}")

        if return_dense:
            output["dense_embeddings"] = results["dense_vecs"]

        if return_sparse:
            output["sparse_embeddings"] = results["lexical_weights"]

        # Check if len(sparse_embeddings) matches len(tokens)

        # if len(output["sparse_embeddings"]) != len(tokens):
        #     raise ValueError("Mismatch between number of sparse embeddings and tokenized inputs.")
        return output


def _to_token_id_weights(
    lexical_weights: dict[str, float], tokenizer
) -> dict[int, float]:
    """
    Convert BGE-M3 lexical_weights to {token_id: weight}.

    IMPORTANT: BGE-M3's lexical_weights returns token IDs as STRING keys
    (e.g., '3350', '5890'), NOT token strings (e.g., '▁kinh', 'doanh').
    We simply convert the string keys to integers directly.

    Args:
        lexical_weights: Dict mapping string token IDs to weights
        tokenizer: BGE-M3 tokenizer (used to filter special tokens)

    Returns:
        Dict mapping integer token IDs to weights
    """
    # Special tokens to filter out
    unused_tokens = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    }

    result = {}
    for token_id_str, weight in lexical_weights.items():
        token_id = int(token_id_str)  # Keys are already token IDs as strings

        if token_id not in unused_tokens and weight > 0:
            # Use max weight if token appears multiple times
            if token_id not in result or weight > result[token_id]:
                result[token_id] = float(weight)

    return result


class VietnameseDenseAdapter(Embeddings):
    """Adapter for VietnameseEmbedding dense embeddings."""

    def __init__(self, vietnamese_instance: VietnameseEmbedding):
        self.vietnamese = vietnamese_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.vietnamese.encode(texts, return_dense=True, return_sparse=False)
        return res["dense_embeddings"]

    def embed_query(self, text: str) -> List[float]:
        res = self.vietnamese.encode([text], return_dense=True, return_sparse=False)
        return res["dense_embeddings"][0]


class VietnameseSparseAdapter(SparseEmbeddings):
    """Adapter for VietnameseEmbedding sparse embeddings for Qdrant."""

    def __init__(self, vietnamese_instance: VietnameseEmbedding):
        self.vietnamese = vietnamese_instance

    def _to_sparse_vector(
        self, token_weights_dict: Dict[str, float]
    ) -> rest.SparseVector:
        """
        Convert BGE-M3 lexical_weights to Qdrant SparseVector.

        Note: BGE-M3 returns keys as string token IDs (e.g., '3350'),
        not token strings (e.g., '▁kinh').
        """
        token_id_weights = _to_token_id_weights(
            token_weights_dict, self.vietnamese.tokenizer
        )

        # Keys are already integers from _to_token_id_weights
        sorted_indices = sorted(token_id_weights.keys())
        values = [token_id_weights[idx] for idx in sorted_indices]

        return rest.SparseVector(indices=sorted_indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[rest.SparseVector]:
        res = self.vietnamese.encode(texts, return_dense=False, return_sparse=True)
        return [self._to_sparse_vector(d) for d in res["sparse_embeddings"]]

    def embed_query(self, text: str) -> rest.SparseVector:
        res = self.vietnamese.encode([text], return_dense=False, return_sparse=True)
        return self._to_sparse_vector(res["sparse_embeddings"][0])


# --- BGE-M3 EMBEDDING (BAAI/bge-m3) ---


class BGEM3Embedding:
    """
    Wrapper for BAAI/bge-m3 multilingual embedding model.

    Uses FlagEmbedding.BGEM3FlagModel for dense + sparse embeddings.
    Supports 100+ languages with 1024-dimensional dense vectors.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device

        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=device == "cuda",
            device=device,
        )
        self.model.model.eval()
        try:
            self.model.encode(["warmup"], batch_size=1, max_length=32)
        except Exception as e:
            print(f"Warmup failed: {e}")
        self.tokenizer = self.model.tokenizer

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        max_length: int = 8192,
        batch_size: int = 16,
    ) -> dict[str, any]:
        """
        Encode texts using BGE-M3 model.

        Returns:
            Dict with "dense_embeddings" and/or "sparse_embeddings" keys.
        """
        if isinstance(texts, str):
            texts = [texts]

        output = {}

        # Ensure eval mode and no_grad for determinism
        self.model.model.eval()
        with torch.no_grad():
            results = self.model.encode(
                texts,
                batch_size=min(batch_size, len(texts)),
                max_length=max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )

        if return_dense:
            output["dense_embeddings"] = results["dense_vecs"]

        if return_sparse:
            output["sparse_embeddings"] = results["lexical_weights"]

        return output


class BGEM3DenseAdapter(Embeddings):
    """Adapter for BGEM3Embedding dense embeddings."""

    def __init__(self, bge_m3_instance: BGEM3Embedding):
        self.bge_m3 = bge_m3_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.bge_m3.encode(texts, return_dense=True, return_sparse=False)
        return res["dense_embeddings"].tolist()

    def embed_query(self, text: str) -> List[float]:
        res = self.bge_m3.encode([text], return_dense=True, return_sparse=False)
        return res["dense_embeddings"][0].tolist()


class BGEM3SparseAdapter(SparseEmbeddings):
    """Adapter for BGEM3Embedding sparse embeddings for Qdrant."""

    def __init__(self, bge_m3_instance: BGEM3Embedding):
        self.bge_m3 = bge_m3_instance

    def _to_sparse_vector(
        self, token_weights_dict: Dict[str, float]
    ) -> rest.SparseVector:
        """
        Convert BGE-M3 lexical_weights to Qdrant SparseVector.

        Note: BGE-M3 returns keys as string token IDs (e.g., '3350'),
        not token strings (e.g., '▁kinh').
        """
        token_id_weights = _to_token_id_weights(
            token_weights_dict, self.bge_m3.tokenizer
        )

        sorted_indices = sorted(token_id_weights.keys())
        values = [token_id_weights[idx] for idx in sorted_indices]

        return rest.SparseVector(indices=sorted_indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[rest.SparseVector]:
        res = self.bge_m3.encode(texts, return_dense=False, return_sparse=True)
        return [self._to_sparse_vector(d) for d in res["sparse_embeddings"]]

    def embed_query(self, text: str) -> rest.SparseVector:
        res = self.bge_m3.encode([text], return_dense=False, return_sparse=True)
        return self._to_sparse_vector(res["sparse_embeddings"][0])
