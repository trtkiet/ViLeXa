import logging
from typing import Optional, Dict, List, Any
import json
import pickle as pkl
from tqdm import tqdm  # Import progress bar
import uuid
from transformers import AutoTokenizer

# LangChain Imports
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    SparseVectorParams, 
    SparseIndexParams, 
    PointStruct,
    SparseVector
)
from collections import defaultdict


from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Config(BaseSettings):
    QDRANT_PORT: int = 6333
    DOCS_ROOT: str = "./law_crawler/vbpl_documents"
    CHUNK_SIZE: int = 1024
    COLLECTION_NAME: str = f"laws"
    DENSE_VECTOR_SIZE: int = 768  # Example size for dense vectors
    MODEL_NAME: str = "Alibaba-NLP/gte-multilingual-base"
    DENSE_EMBEDDINGS_FILE: str = f"data/processed_chunksize_{CHUNK_SIZE}_alibaba/dense_embeddings.pkl"
    SPARSE_EMBEDDINGS_FILE: str = f"data/processed_chunksize_{CHUNK_SIZE}_alibaba/sparse_embeddings.pkl"
    DOCS_FILE: str = f"data/processed_chunksize_{CHUNK_SIZE}_alibaba/documents.json"
    
def _read_embeddings_from_pkl(input_file):
    with open(input_file, 'rb') as f:
        embeddings = pkl.load(f)
    return embeddings
    
def _load_docs_from_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        json_docs = json.load(f)
        docs = [Document(**doc) for doc in json_docs]
    return docs

def to_qdrant_sparse(token_weights_dict, tokenizer):
    """
    Converts a dictionary of {token_string: weight} into a Qdrant SparseVector
    by mapping strings back to integer IDs and ensuring unique indices.
    """
    # Use a dictionary to aggregate weights by index ID
    aggregated_weights = defaultdict(float)

    for token_str, weight in token_weights_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        
        # Ensure we have a valid integer ID
        if isinstance(token_id, int):
            aggregated_weights[token_id] += weight

    # Qdrant requires indices and values to be separate lists
    # It is also good practice (and sometimes required) to keep indices sorted
    sorted_indices = sorted(aggregated_weights.keys())
    values = [aggregated_weights[idx] for idx in sorted_indices]
    
    return models.SparseVector(
        indices=sorted_indices, 
        values=values
    )
    
config = Config()

def ingest_data() -> None:
    client = QdrantClient(host="localhost", port=config.QDRANT_PORT)
    collection_name = config.COLLECTION_NAME
    
    # 1. Handle Collection Recreation
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'.")

    # 2. Create Collection with Hybrid Support
    client.create_collection(
        collection_name=collection_name,
        # Dense vector configuration
        vectors_config={
            "dense": VectorParams(
                size=config.DENSE_VECTOR_SIZE, 
                distance=Distance.COSINE
            )
        },
        # Sparse vector configuration
        sparse_vectors_config={
            "sparse": SparseVectorParams(   
                index=SparseIndexParams(
                    on_disk=False # Useful for large datasets
                )
            )
        }
    )
    
    # 3. Load Data
    docs = _load_docs_from_json(config.DOCS_FILE)
    dense_embeddings = _read_embeddings_from_pkl(config.DENSE_EMBEDDINGS_FILE)
    sparse_embeddings = _read_embeddings_from_pkl(config.SPARSE_EMBEDDINGS_FILE)

    if not (len(docs) == len(dense_embeddings) == len(sparse_embeddings)):
        raise ValueError("Mismatch in length between docs, dense, and sparse embeddings!")

    BATCH_SIZE = 128
    total_docs = len(docs)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # 4. Ingest in Batches
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Uploading Hybrid Points"):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_dense = dense_embeddings[i : i + BATCH_SIZE]
        batch_sparse = sparse_embeddings[i : i + BATCH_SIZE]
        
        points = []
        for doc, dense_vec, sparse_dict in zip(batch_docs, batch_dense, batch_sparse):
            
            # SentenceTransformers return sparse as {token_id: weight}
            # Qdrant expects SparseVector(indices=[...], values=[...])
            sparse_vector = to_qdrant_sparse(sparse_dict, tokenizer)  # Provide tokenizer if needed

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec,
                        "sparse": sparse_vector
                    },
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False 
        )

    logger.info("Hybrid data ingestion completed.")

if __name__ == "__main__":
    ingest_data()