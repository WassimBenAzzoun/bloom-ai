# ==============================
# CPU-only FastAPI image
# ==============================
FROM python:3.11-slim

# ---- system deps (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# ---- working dir
WORKDIR /app

# ---- copy requirements
COPY requirements.txt .

# ---- install python deps (CPU only)
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy app
COPY main.py .
COPY models ./models

# ---- expose API
EXPOSE 8000

# ---- start server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

