# ViLeXa - Vietnamese Legal Advisor System

A RAG-powered (Retrieval-Augmented Generation) legal assistant for Vietnamese law. This system allows users to ask questions about Vietnamese legislation and receive accurate answers with citations to specific legal articles.

## Features

- **RAG-powered Q&A** - Ask questions in natural language, get answers with legal citations
- **User Authentication** - JWT-based auth with session persistence
- **Multiple RAG Pipelines** - GTE, Vietnamese Embedding, BGE-M3, and Agentic workflows
- **Hybrid Retrieval** - Combined dense + sparse vector search for better accuracy
- **Document Browser** - Browse, search, and export law documents as PDF
- **Session Management** - Persistent chat history across sessions

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│    React     │────▶│   FastAPI    │────▶│  Qdrant (Hybrid Retrieval)  │
│   Frontend   │     │   Backend    │     │  Dense + Sparse Vectors     │
└──────────────┘     └──────┬───────┘     └─────────────────────────────┘
                            │                           │
                    ┌───────┴───────┐                   ▼
                    ▼               ▼           ┌───────────────┐
              ┌──────────┐   ┌──────────┐       │   Documents   │
              │  SQLite  │   │  Gemini  │◀───── │   Context     │
              │(Auth/DB) │   │   LLM    │       └───────────────┘
              └──────────┘   └──────────┘
```

### Components

| Component | Technology | Description |
|-----------|------------|-------------|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS 4 | Chat UI with document browser |
| Backend | Python 3.10+ / FastAPI / LangChain | RAG pipeline orchestration |
| Vector DB | Qdrant | Hybrid dense + sparse retrieval |
| Embeddings | Alibaba GTE multilingual (default) | 768-dim dense + sparse vectors |
| LLM | Google Gemini | `gemini-2.5-flash-lite` (configurable) |
| Auth DB | SQLite (SQLAlchemy) | User accounts and chat sessions |

## Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development/preprocessing)
- Node.js 18+ (for frontend development)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Quick Start

### 1. Clone and Setup Environment

```bash
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
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 3. Ingest Law Documents

Before the system can answer questions, ingest the law documents into Qdrant:

```bash
# Install dependencies
pip install -e .

# Run ingestion with Alibaba GTE embeddings (recommended)
python preprocess/ingest_data_alibaba.py
```

This will:
- Load chunked documents from `data/processed_chunksize_1024_alibaba/`
- Create Qdrant collection with hybrid (dense + sparse) vectors
- Batch upload all document embeddings

**Note**: First run will download the embedding model (~1GB). GPU acceleration is used if available.

## Development

### Backend

```bash
cd backend
pip install -e .
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Testing

```bash
# Backend tests
pytest

# Frontend linting
cd frontend && npm run lint
```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Create new user account |
| POST | `/api/v1/auth/login` | Login, returns JWT token |
| GET | `/api/v1/auth/me` | Get current user info (requires auth) |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send message, get RAG response with sources |

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_token>" \
  -d '{"message": "Thời gian thử việc tối đa là bao lâu?", "session_id": "abc123"}'
```

**Example Response:**

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

### Other Endpoints

- **Sessions** (`/api/v1/sessions/*`) - CRUD operations for chat sessions
- **Documents** (`/api/v1/documents/*`) - Browse, search, and retrieve law documents

## RAG Pipelines

The system supports multiple RAG pipeline implementations:

| Pipeline | Model | Description |
|----------|-------|-------------|
| **GTE Pipeline** (default) | `Alibaba-NLP/gte-multilingual-base` | Hybrid retrieval with optional reranking |
| **Vietnamese Pipeline** | `AITeamVN/Vietnamese_Embedding_v2` | Optimized for Vietnamese text |
| **BGE-M3 Pipeline** | `BAAI/bge-m3` | Multilingual BGE model |
| **Agentic Pipeline** | LangGraph | Query routing, document grading, adaptive retry |

Configure via environment variable:
```bash
EMBEDDING_MODEL_NAME=Alibaba-NLP/gte-multilingual-base  # default
```

## Benchmarks & Evaluation

The system includes a comprehensive evaluation framework for measuring retrieval and answer quality.

```bash
# Run retrieval benchmarks
python -m benchmarks.run_eval --mode pipeline --collection laws

# Run DeepEval RAG evaluation (requires deepeval)
deepeval test run evaluate/eval.py
```

See [`benchmarks/README.md`](benchmarks/README.md) for detailed documentation on metrics, datasets, and evaluation options.

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *required* | Google Gemini API key |
| `MODEL` | `gemini-2.5-flash-lite` | LLM model name |
| `QDRANT_HOST` | `qdrant` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `DATABASE_URL` | `sqlite:///./law.db` | SQLAlchemy database URL |
| `COLLECTION_NAME` | `laws` | Qdrant collection name |
| `EMBEDDING_MODEL_NAME` | `Alibaba-NLP/gte-multilingual-base` | Embedding model |
| `RETRIEVAL_MODE` | `hybrid` | Retrieval: `hybrid`, `dense`, `sparse` |
| `RETRIEVAL_K` | `10` | Number of candidates to retrieve |
| `TOP_K` | `3` | Final docs sent to LLM |
| `JWT_SECRET_KEY` | *change in prod* | JWT signing secret |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | Token expiry (7 days) |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend API URL |

## Project Structure

```
VietnameseLaw/
├── backend/                     # FastAPI backend
│   ├── api/v1/                 # Route handlers (auth, chat, sessions, lookup)
│   ├── core/                   # Config, database, security
│   ├── db/models/              # SQLAlchemy models
│   ├── models/                 # Pydantic schemas
│   └── services/               # Business logic
│       ├── pipelines/          # RAG pipeline implementations
│       └── rerankers/          # Cross-encoder rerankers
├── frontend/                    # React frontend
│   └── src/
│       ├── components/         # UI components
│       ├── contexts/           # React contexts (auth)
│       ├── hooks/              # Custom hooks
│       ├── pages/              # Page components
│       ├── services/           # API clients
│       └── types/              # TypeScript types
├── benchmarks/                  # Evaluation framework
│   ├── run_eval.py             # Retrieval benchmarks
│   └── run_deepeval.py         # DeepEval RAG evaluation
├── preprocess/                  # Data ingestion scripts
│   ├── chunker.py              # Hierarchical document chunking
│   ├── ingest_data_alibaba.py  # GTE embedding ingestion
│   └── ingest_data_vietnamese.py
├── data/                        # Processed embeddings & documents
├── law_crawler/                 # Source law documents
│   └── vbpl_documents/         # Crawled JSON documents
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

## Troubleshooting

### Qdrant Connection Issues

Ensure Qdrant is running and accessible:
```bash
curl http://localhost:6333/healthz
```

### Model Download Issues

The embedding model is downloaded from Hugging Face on first run. Ensure you have internet access and sufficient disk space (~1GB).

### Docker Memory Issues

Increase Docker memory allocation if containers crash:
- Docker Desktop → Settings → Resources → Memory: 4GB+

### GPU Not Detected

For GPU acceleration, ensure NVIDIA drivers and CUDA are installed. The backend container is configured for GPU passthrough.

## License

See [LICENSE](LICENSE) for details.
