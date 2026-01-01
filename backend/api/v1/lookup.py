"""
Law Document Lookup API endpoints.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from models.lookup import (
    DocumentResponse,
    DocumentListResponse,
    ArticleResponse,
    TypesResponse,
)
from services.lookup_service import LawDocumentService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_lookup_service() -> LawDocumentService:
    """Dependency to get the shared lookup service instance."""
    from main import lookup_service
    return lookup_service


@router.get("/types", response_model=TypesResponse)
async def get_document_types(
    service: LawDocumentService = Depends(get_lookup_service)
):
    """Get list of all available document types."""
    logger.info("Fetching document types")
    types = service.get_types()
    logger.info(f"Found {len(types)} document types")
    return TypesResponse(types=types)


@router.get("/search", response_model=DocumentListResponse)
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    type: Optional[str] = Query(None, description="Filter by document type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    service: LawDocumentService = Depends(get_lookup_service)
):
    """
    Search for documents matching the query.
    Uses simple text-based search in title and content.
    """
    logger.info(f"Searching documents: query='{q}', type={type}, page={page}")
    result = service.search_documents(
        query=q,
        doc_type=type,
        page=page,
        page_size=page_size
    )
    logger.info(f"Search returned {result['total']} results")
    return DocumentListResponse(**result)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    type: Optional[str] = Query(None, description="Filter by document type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    service: LawDocumentService = Depends(get_lookup_service)
):
    """
    List all documents with optional filtering and pagination.
    """
    logger.info(f"Listing documents: type={type}, page={page}, page_size={page_size}")
    result = service.list_documents(
        doc_type=type,
        page=page,
        page_size=page_size
    )
    logger.info(f"Listed {len(result['items'])} of {result['total']} documents")
    return DocumentListResponse(**result)


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    service: LawDocumentService = Depends(get_lookup_service)
):
    """
    Get a single document by ID with full content.
    Accepts both pure ID (e.g., '11716') and filename patterns.
    """
    logger.info(f"Getting document: {doc_id}")
    document = service.get_document(doc_id)
    if not document:
        logger.warning(f"Document not found: {doc_id}")
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info(f"Retrieved document: {document['title']}")
    return DocumentResponse(**document)


@router.get("/{doc_id}/articles", response_model=List[ArticleResponse])
async def get_document_articles(
    doc_id: str,
    service: LawDocumentService = Depends(get_lookup_service)
):
    """
    Get all articles from a document.
    Returns parsed articles with chapter/section context.
    """
    logger.info(f"Getting articles for document: {doc_id}")
    articles = service.get_articles(doc_id)
    if not articles:
        logger.warning(f"No articles found for document: {doc_id}")
        raise HTTPException(status_code=404, detail="Document not found or has no articles")
    logger.info(f"Retrieved {len(articles)} articles for document: {doc_id}")
    return [ArticleResponse(**article) for article in articles]
