# CPU image serving CatBoost (Part A) and the ONNX FT-Transformer (Part B).
# Build from the repository root (models/ must exist, see README):
#   docker build -f api/Dockerfile -t p2-api .
#   docker run -p 8000:8000 p2-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODELS_DIR=/app/models ORT_THREADS=1
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/__init__.py src/__init__.py
COPY src/part_b/__init__.py src/part_b/preprocessing.py src/part_b/
COPY api/__init__.py api/main.py api/
COPY models/ models/

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
