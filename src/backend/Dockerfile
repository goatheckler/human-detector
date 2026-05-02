FROM python:3.14-slim@sha256:5b3879b6f3cb77e712644d50262d05a7c146b7312d784a18eff7ff5462e77033

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/
COPY src/backend/ ./src/backend/

ENV PYTHONPATH=/app
# YOLO model size: yolo11n.pt (fastest), yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt (most accurate)
ENV HUMAN_DETECTOR_MODEL_SIZE=yolo11n.pt
# Detection confidence threshold (0.0-1.0)
ENV HUMAN_DETECTOR_CONFIDENCE_THRESHOLD=0.45
# Supported devices: ["cpu","gpu"] or ["cpu"] or ["gpu"]
ENV HUMAN_DETECTOR_SUPPORTED_DEVICES='["cpu","gpu"]'
# CPU thread count (1-64)
ENV HUMAN_DETECTOR_CPU_THREADS=32

EXPOSE 8000

CMD ["uvicorn", "src.backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
