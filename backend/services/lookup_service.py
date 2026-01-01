"""
Law Document Lookup Service

Provides document listing, search, and retrieval functionality
for the Vietnamese Law lookup page.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from core.config import config

logger = logging.getLogger(__name__)


@dataclass
class LawDocument:
    """Represents a formatted law document for the lookup page"""
    id: str
    title: str
    type: str
    number: str
    date: str
    snippet: str
    content: str
    category: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LawArticle:
    """Represents a single article within a law"""
    article_number: str
    article_title: str
    content: str
    chapter: Optional[str] = None
    section: Optional[str] = None


def parse_filename(filepath: str) -> Dict[str, str]:
    """
    Parse the filename to extract metadata.
    Format: {id}_{type} {number}.json
    Example: 11716_Luật 36_2009_QH12.json
    """
    filename = Path(filepath).stem
    category = Path(filepath).parent.name

    parts = filename.split('_', 1)
    doc_id = parts[0]

    if len(parts) > 1:
        rest = parts[1]
        type_match = re.match(
            r'^(Luật|Bộ luật|Nghị định|Nghị quyết|Thông tư|Pháp lệnh|Quyết định|Hiến pháp|Thông tư liên tịch)',
            rest
        )
        if type_match:
            doc_type = type_match.group(1)
            number_part = rest[len(doc_type):].strip()
            doc_number = number_part.replace('_', '/')
        else:
            doc_type = category.replace('_', ' ')
            doc_number = rest.replace('_', '/')
    else:
        doc_type = category.replace('_', ' ')
        doc_number = ""

    year_match = re.search(r'/(\d{4})/', doc_number)
    year = year_match.group(1) if year_match else ""

    return {
        "id": doc_id,
        "type": doc_type,
        "number": doc_number,
        "year": year,
        "category": category,
        "title": f"{doc_type} {doc_number}".strip()
    }


def extract_title_from_content(content: str) -> str:
    """
    Extract the title from the content.
    The title is usually at the beginning, after the document type.
    """
    lines = content.strip().split('\n')
    title_parts = []
    found_type = False

    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue

        if line.upper() in ['LUẬT', 'BỘ LUẬT', 'NGHỊ ĐỊNH', 'THÔNG TƯ', 
                            'QUYẾT ĐỊNH', 'PHÁP LỆNH', 'NGHỊ QUYẾT', 'HIẾN PHÁP']:
            found_type = True
            continue

        if line.startswith('Căn cứ') or line.startswith('CHƯƠNG') or line.startswith('Điều 1'):
            break

        if line.startswith('_____') or line.startswith('-----'):
            break

        if found_type and line:
            title_parts.append(line)
            if len(title_parts) >= 2:
                break

    return ' '.join(title_parts).strip() if title_parts else ""


def create_snippet(content: str, max_length: int = 200) -> str:
    """Create a short snippet from the content for preview."""
    text = content.strip()

    article_match = re.search(r'(Điều\s+\d+\..*?)(?=Điều\s+\d+\.|$)', text, re.DOTALL)
    if article_match:
        snippet_text = article_match.group(1).strip()
    else:
        snippet_text = text[:500]

    snippet_text = ' '.join(snippet_text.split())

    if len(snippet_text) > max_length:
        snippet_text = snippet_text[:max_length].rsplit(' ', 1)[0] + '...'

    return snippet_text


def parse_articles(content: str) -> List[LawArticle]:
    """
    Parse the content to extract individual articles.
    Uses regex to split by 'Điều X.' pattern.
    """
    articles = []

    article_pattern = re.compile(r'(Điều\s+\d+[a-z]?\..*?)(?=Điều\s+\d+[a-z]?\.|$)', re.DOTALL)
    chapter_pattern = re.compile(r'(CHƯƠNG\s+[IVX\d]+[^a-z]*)', re.MULTILINE)
    section_pattern = re.compile(r'(Mục\s+\d+\.?[^\n]*)', re.MULTILINE)

    chapters = [(m.group(1).strip(), m.start()) for m in chapter_pattern.finditer(content)]
    sections = [(m.group(1).strip(), m.start()) for m in section_pattern.finditer(content)]

    for match in article_pattern.finditer(content):
        article_text = match.group(1).strip()
        article_start = match.start()

        num_match = re.match(r'(Điều\s+\d+[a-z]?)', article_text)
        article_number = num_match.group(1) if num_match else ""

        first_line = article_text.split('\n')[0].strip()

        current_chapter = None
        for ch_name, ch_pos in reversed(chapters):
            if ch_pos < article_start:
                current_chapter = ch_name
                break

        current_section = None
        for sec_name, sec_pos in reversed(sections):
            if sec_pos < article_start:
                current_section = sec_name
                break

        articles.append(LawArticle(
            article_number=article_number,
            article_title=first_line,
            content=article_text,
            chapter=current_chapter,
            section=current_section
        ))

    return articles


def load_law_document(filepath: str) -> LawDocument:
    """
    Load and parse a law document from a JSON file.
    Returns a LawDocument object ready for the frontend.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = parse_filename(filepath)
    content_title = extract_title_from_content(data['Content'])

    if content_title:
        title = f"{meta['type']} {meta['number']} - {content_title}"
    else:
        title = meta['title']

    snippet = create_snippet(data['Content'])

    return LawDocument(
        id=data['Id'],
        title=title,
        type=meta['type'],
        number=meta['number'],
        date=meta['year'],
        snippet=snippet,
        content=data['Content'],
        category=meta['category']
    )


def format_for_frontend(doc: LawDocument) -> Dict:
    """
    Format a LawDocument for the frontend API response.
    Matches the LawDocument interface in frontend/src/types/index.ts
    """
    return {
        "id": doc.id,
        "title": doc.title,
        "type": doc.type,
        "ref": f"{doc.type} {doc.number}",
        "date": doc.date,
        "snippet": doc.snippet,
        "content": doc.content
    }


class LawDocumentService:
    """
    Service class for law document operations.
    Uses Qdrant for fast semantic search and file system for document retrieval.
    """

    def __init__(self, docs_root: Optional[str] = None):
        self.docs_root = Path(docs_root) if docs_root else Path(config.DOCS_ROOT)
        self._id_to_filepath: Dict[str, str] = {}
        self._qdrant: Optional["QdrantClient"] = None
        self._embedder: Optional["SentenceTransformer"] = None
        self._initialized = False

    def startup(self) -> None:
        """Initialize the service: build ID mapping and connect to Qdrant."""
        logger.info("Initializing LawDocumentService...")
        self._build_id_mapping()
        self._init_qdrant()
        logger.info("LawDocumentService initialized successfully")

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client and embedding model for semantic search."""
        if self._initialized:
            return

        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            # Try Docker hostname first, fall back to localhost
            try:
                self._qdrant = QdrantClient(host="qdrant", port=config.QDRANT_PORT)
                # Test connection
                self._qdrant.get_collections()
                logger.info("Connected to Qdrant at 'qdrant' hostname")
            except Exception:
                self._qdrant = QdrantClient(host="localhost", port=config.QDRANT_PORT)
                logger.info("Connected to Qdrant at 'localhost'")

            self._embedder = SentenceTransformer('minhquan6203/paraphrase-vietnamese-law')
            logger.info("Loaded embedding model for semantic search")
            self._initialized = True

        except Exception as e:
            logger.warning(f"Could not initialize Qdrant/Embedder: {e}. Falling back to text search.")
            self._qdrant = None
            self._embedder = None
            self._initialized = True

    def _get_cache_path(self) -> Path:
        """Get the path to the cache file."""
        return self.docs_root / ".id_mapping_cache.json"

    def _load_cache(self) -> bool:
        """Load ID mapping from cache file. Returns True if cache was loaded."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache has expected structure
            if not isinstance(cache_data, dict) or 'mapping' not in cache_data:
                logger.warning("Invalid cache format, rebuilding...")
                return False

            self._id_to_filepath = cache_data['mapping']
            logger.info(f"Loaded {len(self._id_to_filepath)} documents from cache")
            return True

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save ID mapping to cache file."""
        cache_path = self._get_cache_path()
        try:
            cache_data = {
                'mapping': self._id_to_filepath,
                'doc_count': len(self._id_to_filepath)
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved cache with {len(self._id_to_filepath)} documents")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _build_id_mapping(self, force_rebuild: bool = False) -> None:
        """Build mapping from document ID to filepath."""
        if self._id_to_filepath and not force_rebuild:
            return

        # Try loading from cache first
        if not force_rebuild and self._load_cache():
            return
        
        logger.info(f"Building document ID mapping from {self.docs_root}")
        self._id_to_filepath = {}
        for filepath in self.docs_root.glob("**/*.json"):
            # Skip the cache file itself
            if filepath.name == ".id_mapping_cache.json":
                continue
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_id = str(data.get('Id', ''))
                    if doc_id:
                        self._id_to_filepath[doc_id] = str(filepath)
            except Exception as e:
                logger.warning(f"Failed to index {filepath}: {e}")
                continue
        logger.info(f"Indexed {len(self._id_to_filepath)} documents")

        # Save to cache for next startup
        self._save_cache()

    def rebuild_cache(self) -> None:
        """Force rebuild the document ID mapping cache."""
        logger.info("Force rebuilding document cache...")
        self._build_id_mapping(force_rebuild=True)

    def _find_filepath_by_id(self, doc_id: str) -> Optional[str]:
        """
        Find filepath by document ID.
        Handles both pure ID (e.g., "11716") and filename patterns.
        """
        self._build_id_mapping()
        
        # Direct ID match
        if doc_id in self._id_to_filepath:
            return self._id_to_filepath[doc_id]
        
        # Try to extract ID from filename pattern (e.g., "11716_Luật...")
        id_part = doc_id.split('_')[0] if '_' in doc_id else doc_id
        if id_part in self._id_to_filepath:
            return self._id_to_filepath[id_part]
        
        return None

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get a single document by ID with full content."""
        logger.debug(f"Getting document by ID: {doc_id}")
        filepath = self._find_filepath_by_id(doc_id)
        if not filepath:
            logger.warning(f"Document not found: {doc_id}")
            return None

        try:
            doc = load_law_document(filepath)
            logger.info(f"Loaded document: {doc.title}")
            return format_for_frontend(doc)
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None

    def get_articles(self, doc_id: str) -> List[Dict]:
        """Get all articles from a document."""
        filepath = self._find_filepath_by_id(doc_id)
        if not filepath:
            return []

        try:
            doc = load_law_document(filepath)
            articles = parse_articles(doc.content)
            return [asdict(a) for a in articles]
        except Exception:
            return []

    def list_documents(
        self,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """List documents with optional filtering and pagination."""
        documents = []

        for filepath in self.docs_root.glob("**/*.json"):
            try:
                doc = load_law_document(str(filepath))
                
                if doc_type and doc.type != doc_type:
                    continue

                documents.append({
                    "id": doc.id,
                    "title": doc.title,
                    "type": doc.type,
                    "ref": f"{doc.type} {doc.number}",
                    "date": doc.date,
                    "snippet": doc.snippet
                })
            except Exception:
                continue

        # Sort by date descending
        documents.sort(key=lambda x: x.get('date', ''), reverse=True)

        # Paginate
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        items = documents[start:end]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
        }

    def search_documents(
        self,
        query: str,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """
        Search for documents matching the query.
        Uses Qdrant semantic search for fast results, falls back to text search.
        """
        self._init_qdrant()

        # Try semantic search with Qdrant
        if self._qdrant and self._embedder:
            return self._search_with_qdrant(query, doc_type, page, page_size)
        
        # Fallback to text search
        logger.info("Using fallback text search")
        return self._search_with_text(query, doc_type, page, page_size)

    def _search_with_qdrant(
        self,
        query: str,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """
        Semantic search using Qdrant vector database.
        Returns documents ranked by semantic similarity.
        """
        try:
            # Encode query to vector
            query_vector = self._embedder.encode(query).tolist()
            
            # Search with larger limit to allow for deduplication and filtering
            search_limit = page_size * 5  # Get more results for deduplication
            
            logger.info(f"Searching Qdrant for: '{query}'")
            search_results = self._qdrant.query_points(
                collection_name="laws",
                query=query_vector,
                limit=search_limit,
                with_payload=True
            )

            points = search_results.points if hasattr(search_results, 'points') else []
            logger.info(f"Qdrant returned {len(points)} results")

            # Deduplicate by law_id and collect unique documents
            seen_law_ids = set()
            unique_docs = []

            for point in points:
                payload = point.payload
                law_id = str(payload.get('law_id', ''))
                
                if not law_id or law_id in seen_law_ids:
                    continue
                
                seen_law_ids.add(law_id)

                # Load full document to get metadata
                filepath = self._find_filepath_by_id(law_id)
                if not filepath:
                    continue

                try:
                    doc = load_law_document(filepath)
                    
                    # Filter by document type if specified
                    if doc_type and doc.type != doc_type:
                        continue

                    # Create snippet from matched article if available
                    matched_snippet = payload.get('source_text', '')[:200]
                    if matched_snippet:
                        matched_snippet = matched_snippet.strip() + '...'

                    unique_docs.append({
                        "id": doc.id,
                        "title": doc.title,
                        "type": doc.type,
                        "ref": f"{doc.type} {doc.number}",
                        "date": doc.date,
                        "snippet": matched_snippet or doc.snippet,
                        "score": point.score if hasattr(point, 'score') else 0
                    })
                except Exception as e:
                    logger.warning(f"Error loading document {law_id}: {e}")
                    continue

            # Sort by relevance score (already sorted by Qdrant, but ensure order)
            unique_docs.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Remove score from final output
            for doc in unique_docs:
                doc.pop('score', None)

            # Paginate
            total = len(unique_docs)
            start = (page - 1) * page_size
            end = start + page_size
            items = unique_docs[start:end]

            logger.info(f"Returning {len(items)} unique documents from Qdrant search")

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
            }

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}. Falling back to text search.")
            return self._search_with_text(query, doc_type, page, page_size)

    def _search_with_text(
        self,
        query: str,
        doc_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """
        Fallback text-based search (case-insensitive).
        Used when Qdrant is not available.
        """
        results = []
        query_lower = query.lower()

        for filepath in self.docs_root.glob("**/*.json"):
            try:
                doc = load_law_document(str(filepath))

                if doc_type and doc.type != doc_type:
                    continue

                # Simple text search in title and content
                if query_lower in doc.title.lower() or query_lower in doc.content.lower():
                    results.append({
                        "id": doc.id,
                        "title": doc.title,
                        "type": doc.type,
                        "ref": f"{doc.type} {doc.number}",
                        "date": doc.date,
                        "snippet": doc.snippet
                    })
            except Exception:
                continue

        # Sort by date descending
        results.sort(key=lambda x: x.get('date', ''), reverse=True)

        # Paginate
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        items = results[start:end]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
        }

    def get_types(self) -> List[str]:
        """Get list of all document types."""
        types = set()

        for filepath in self.docs_root.glob("**/*.json"):
            try:
                doc = load_law_document(str(filepath))
                types.add(doc.type)
            except Exception:
                continue

        return sorted(types)
