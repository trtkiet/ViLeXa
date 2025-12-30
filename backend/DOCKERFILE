FROM python:3.10 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /backend

# Install runtime deps
RUN python -m pip install --upgrade --no-cache-dir pip

# Copy dependency metadata first for better layer caching
# COPY pyproject.toml .

# Install app requirements (no build tooling needed for current deps)
RUN python -m pip install --no-cache-dir \
	"fastapi[standard]" \
	"google-genai" \
	"httpx" \
	"uvicorn"\
	"qdrant-client" \
	"sentence-transformers"

# Copy backend source
COPY . .

EXPOSE 8000

# GEMINI_API_KEY must be provided at runtime
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
