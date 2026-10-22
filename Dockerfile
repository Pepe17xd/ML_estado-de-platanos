FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/production_model_v4.keras \
    METADATA_PATH=/app/models/metadata.json

WORKDIR /app
COPY requirements-linux-amd64.lock.txt ./requirements.lock.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.lock.txt \
    && groupadd --gid 10001 banana \
    && useradd --uid 10001 --gid banana --create-home banana

COPY api/__init__.py ./api/__init__.py
COPY api/main.py ./api/main.py
COPY api/routes.py ./api/routes.py
COPY api/schemas.py ./api/schemas.py
COPY inference/__init__.py ./inference/__init__.py
COPY inference/image_quality.py ./inference/image_quality.py
COPY inference/model_loader.py ./inference/model_loader.py
COPY inference/predictor.py ./inference/predictor.py
COPY inference/preprocessing.py ./inference/preprocessing.py
COPY training/__init__.py ./training/__init__.py
COPY training/v3_e1/__init__.py ./training/v3_e1/__init__.py
COPY training/v3_e1/augmentations.py ./training/v3_e1/augmentations.py
COPY models/production_model_v4.keras ./models/production_model_v4.keras
COPY models/metadata.json ./models/metadata.json

RUN chown -R banana:banana /app
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
