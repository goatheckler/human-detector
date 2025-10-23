# 🚶 Human Detector

YOLO11-powered human detection API with FastAPI backend and Streamlit UI.

**🌐 Try it live:** https://humans.goatheckler.com

## ✨ Features

- 🎯 Real-time human detection using YOLO11
- 🖼️ Streamlit web UI for image uploads
- 🚀 FastAPI REST API for programmatic access
- 📦 Docker containerized deployment
- ⚡ CPU and GPU support
- 🎚️ Configurable confidence thresholds and model sizes

## 🐳 Quick Start (Docker)

```bash
docker compose up -d
```

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

Models download automatically on first startup (~5MB, <1s).

## 🛠️ Development Setup

### Backend

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run-backend.sh
```

### Frontend

```bash
cd src/frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run-frontend.sh
```

### Tests

```bash
cd src/tests
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

## 📡 API Usage

### POST /detect

**Request:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@photo.jpg" \
  -F "device=cpu" \
  -F "cpu_threads=4"
```

**Response:**
```json
{
  "humanDetected": true,
  "boundingBoxes": [
    {
      "x1": 47.4,
      "y1": 75.5,
      "x2": 184.4,
      "y2": 232.4,
      "confidence": 0.84
    }
  ],
  "maxConfidence": 0.84
}
```

## ⚙️ Configuration

Environment variables (Docker):

```yaml
# Backend
HUMAN_DETECTOR_MODEL_SIZE=yolo11n.pt          # n, s, m, l, x
HUMAN_DETECTOR_CONFIDENCE_THRESHOLD=0.45       # 0.0-1.0
HUMAN_DETECTOR_SUPPORTED_DEVICES=["cpu"]       # cpu, gpu
HUMAN_DETECTOR_CPU_THREADS=32                  # 1-64

# Frontend
API_BASE_URL=http://backend:8000
UI_CPU_THREADS_MIN=1
UI_CPU_THREADS_MAX=64
UI_CPU_THREADS_DEFAULT=32
```

## 📁 Project Structure

```
src/
├── backend/
│   ├── api/          # FastAPI endpoints
│   ├── models/       # Pydantic models
│   ├── services/     # Detection service
│   └── config.py     # Settings
├── frontend/
│   └── app.py        # Streamlit UI
└── tests/
    ├── api/          # API tests
    ├── models/       # Model tests
    ├── services/     # Service tests
    ├── integration/  # Integration tests
    └── fixtures/     # Test images
```

## 📝 License

MIT
