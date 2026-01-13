# Use an official PyTorch image with CUDA support
# This image includes Python (usually 3.10 or 3.11) and the GPU drivers for PyTorch
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /backend

# Install system dependencies if your specific python packages need C++ build tools
# (Optional, but often needed for specific libraries)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend source
COPY . .

# Install app requirements
# NOTE: If requirements.txt contains 'torch', pip might try to reinstall it.
# It is faster to remove 'torch' from requirements.txt or run this:
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

EXPOSE 8000

# GEMINI_API_KEY must be provided at runtime
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]