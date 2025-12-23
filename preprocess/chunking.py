import json
import re
import os
import uuid
import glob
from typing import List, Dict, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

@dataclass
class LawChunk:
    law_id: str
    text: str
    metadata: Dict[str, str]

class VietnameseLawChunker:
    def __init__(self):
        # Regex patterns
        self.chapter_pattern = re.compile(r'(^CHƯƠNG\s+[IVX]+.*?$)', re.MULTILINE)
        self.section_pattern = re.compile(r'(^Mục\s+\d+.*?$)', re.MULTILINE)
        self.article_pattern = re.compile(r'(^Điều\s+\d+\..*?$)', re.MULTILINE)
        self.clause_pattern = re.compile(r'(^\d+\.\s+)', re.MULTILINE)
        self.point_pattern = re.compile(r'(^[a-z]\)\s+)', re.MULTILINE)

    def parse_file(self, file_path: str) -> List[LawChunk]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        law_id = data.get('Id', '')
        content = data.get('Content', '')
        
        return self.chunk_text(law_id, content)

    def chunk_text(self, law_id: str, text: str) -> List[LawChunk]:
        chunks = []
        
        # Split by Chapter
        chapters = self._split_keep_delimiter(text, self.chapter_pattern)
        
        current_chapter = ""
        
        for chapter_text in chapters:
            if self.chapter_pattern.match(chapter_text):
                current_chapter = chapter_text.split('\n')[0].strip()
            
            # Inside a chapter (or at start), split by Section (Mục)
            sections = self._split_keep_delimiter(chapter_text, self.section_pattern)
            
            current_section = ""
            
            for section_text in sections:
                if self.section_pattern.match(section_text):
                    current_section = section_text.split('\n')[0].strip()
                
                # Inside section (or chapter), split by Article (Điều)
                articles = self._split_keep_delimiter(section_text, self.article_pattern)
                
                for article_text in articles:
                    if not self.article_pattern.match(article_text):
                        continue 
                    
                    # Extract Article Title
                    lines = article_text.split('\n')
                    article_header = lines[0].strip()
                    
                    # Extract Article Number
                    article_number_match = re.match(r'(Điều\s+\d+)', article_header)
                    article_number = article_number_match.group(1) if article_number_match else article_header
                    
                    # Split by Clause (Khoản)
                    clauses = re.split(r'(?=\n\d+\.\s+)', article_text)
                    
                    for i, clause_text in enumerate(clauses):
                        clause_text = clause_text.strip()
                        if not clause_text: continue
                        
                        clause_match = re.match(r'^(\d+)\.\s+', clause_text)
                        
                        if clause_match:
                            clause_number = clause_match.group(1)
                        else:
                            if article_header in clause_text:
                                clause_number = "0"
                            else:
                                clause_number = f"p_{i}"
                        
                        metadata = {
                            "law_id": law_id,
                            "chapter": current_chapter,
                            "section": current_section,
                            "article": article_number,
                            "article_title": article_header,
                            "clause": clause_number,
                            "source_text": article_text
                        }
                        x
                        
        return chunks

    def _split_keep_delimiter(self, text: str, pattern) -> List[str]:
        parts = pattern.split(text)
        result = []
        if parts[0].strip():
            result.append(parts[0])
            
        for i in range(1, len(parts), 2):
            delimiter = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            result.append(delimiter + content)
            
        return result

class LawIngestor:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "laws"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            'minhquan6203/paraphrase-vietnamese-law',
            device = "cuda" if torch.cuda.is_available() else "cpu")
        self._setup_collection()

    def _setup_collection(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
        else:
            print(f"Collection '{self.collection_name}' exists.")

    def ingest_file(self, file_path: str):
        chunker = VietnameseLawChunker()
        try:
            chunks = chunker.parse_file(file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return

        if not chunks:
            print(f"No chunks found in {file_path}")
            return

        print(f"Embedding {len(chunks)} chunks from {os.path.basename(file_path)}...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunk.text,
                    **chunk.metadata
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points.")

if __name__ == "__main__":
    # Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Adjust path to point to the correct directory relative to this script
    DOCS_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "law_crawler", "vbpl_documents")
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    ingestor = LawIngestor(qdrant_url=QDRANT_URL)
    
    # Ingest all files in directory recursively
    json_files = glob.glob(os.path.join(DOCS_ROOT_DIR, "**", "*.json"), recursive=True)
    print(f"Found {len(json_files)} law documents in {DOCS_ROOT_DIR}")
    
    # Process all files
    for file_path in json_files: 
        ingestor.ingest_file(file_path)
