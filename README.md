# Vietnamese Law Lookup and Advising System

A RAG-powered (Retrieval-Augmented Generation) legal assistant for Vietnamese law. This system allows users to ask questions about Vietnamese legislation and receive accurate answers with citations to specific legal articles.

## Architecture

```
User → Frontend (React) → Backend (FastAPI) → RAG Pipeline → Gemini LLM
                                    ↓
                              Qdrant (Vector DB)
                                    ↑
                         Preprocessed Law Documents
```

### Components

- **Frontend**: React + TypeScript + Vite with a Gemini-style chat UI
- **Backend**: FastAPI with RAG integration
- **Vector Database**: Qdrant for semantic search
- **Embedding Model**: `minhquan6203/paraphrase-vietnamese-law`
- **LLM**: Google Gemini 1.5 Pro
- **Data Source**: Vietnamese legal documents from VBPL (vbpl_documents)

## Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for preprocessing)
- Node.js 18+ (for frontend development)
- A Gemini API key from Google AI Studio

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd VietnameseLaw

# Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 2. Start Services with Docker

```bash
# Start all services (backend, frontend, qdrant)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

Services will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 3. Run Preprocessing (Data Ingestion)

Before the system can answer questions, you need to ingest the law documents into Qdrant.

```bash
# Install preprocessing dependencies
pip install -r preprocess/requirements.txt

# Run the ingestion script
python preprocess/chunking.py
```

This will:
- Parse all JSON files in `law_crawler/vbpl_documents/`
- Split documents hierarchically (Chương → Mục → Điều → Khoản)
- Generate embeddings using the Vietnamese law model
- Upload vectors to Qdrant

**Note**: First run will download the embedding model (~1GB). GPU acceleration is used if available.

## Project Structure

```
VietnameseLaw/
├── backend/                 # FastAPI backend
│   ├── api/v1/             # API routes
│   ├── services/           # Business logic (chat_service.py)
│   ├── models/             # Pydantic models
│   └── core/               # Configuration
├── frontend/               # React frontend
│   └── src/
│       ├── pages/          # ChatPage, LookupPage
│       ├── components/     # UI components
│       └── types/          # TypeScript interfaces
├── preprocess/             # Data preprocessing
│   └── chunking.py         # Hierarchical chunking & ingestion
├── law_crawler/            # Law document crawler
│   └── vbpl_documents/     # Crawled JSON documents
│       ├── Luật/
│       ├── Bộ_luật/
│       ├── Hiến_pháp/
│       └── Quyết_định/
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

## Preprocessing Details

The preprocessing script (`preprocess/chunking.py`) implements hierarchical chunking following Vietnamese legal document structure:

| Level | Vietnamese | Description |
|-------|------------|-------------|
| 1 | Chương | Chapter |
| 2 | Mục | Section |
| 3 | Điều | Article |
| 4 | Khoản | Clause |
| 5 | Điểm | Point |

Each chunk is stored with metadata for precise citations:
```json
{
  "law_id": "101873",
  "chapter": "CHƯƠNG I",
  "section": "Mục 1",
  "article": "Điều 5",
  "article_title": "Điều 5. Thông tin công dân được tiếp cận",
  "clause": "1",
  "source_text": "Full article text for context..."
}
```

### Preprocessing Options

```bash
# Set custom Qdrant URL
QDRANT_URL=http://custom-host:6333 python preprocess/chunking.py

# The script processes all document types:
# - Luật (Laws)
# - Bộ luật (Codes)
# - Hiến pháp (Constitution)
# - Quyết định (Decisions)
```

## Development

### Backend Only

```bash
cd backend
pip install -e .
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Only

```bash
cd frontend
npm install
npm run dev
```

### Adding New Law Documents

1. Add JSON files to `law_crawler/vbpl_documents/<type>/`
2. Re-run the preprocessing script:
   ```bash
   python preprocess/chunking.py
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send a question and receive an answer with citations |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Thời gian thử việc tối đa là bao lâu?"}'
```

### Example Response

```json
{
  "reply": "Theo quy định tại Điều 25 Bộ luật Lao động...",
  "sources": [
    {
      "law_id": "96172",
      "article": "Điều 25",
      "article_title": "Điều 25. Thời gian thử việc",
      "clause": "1",
      "source_text": "..."
    }
  ]
}
```

## Troubleshooting

### Qdrant Timeout During Preprocessing

If you encounter timeout errors, the script already handles this with:
- 120-second timeout
- Batch processing (50 points per batch)
- Error recovery per batch

### Model Download Issues

The embedding model is downloaded from Hugging Face on first run. Ensure you have internet access and sufficient disk space (~1GB).

### Docker Memory Issues

Increase Docker memory allocation if containers crash:
- Docker Desktop → Settings → Resources → Memory: 4GB+
