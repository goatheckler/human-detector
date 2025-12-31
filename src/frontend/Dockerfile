FROM python:3.14-slim@sha256:f7864aa85847985ba72d2dcbcbafd7475354c848e1abbdf84f523a100800ae0b

WORKDIR /app

COPY src/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY src/__init__.py ./src/
COPY src/frontend/ ./src/frontend/

ENV API_BASE_URL=http://localhost:8000
ENV UI_CPU_THREADS_MIN=1
ENV UI_CPU_THREADS_MAX=64
ENV UI_CPU_THREADS_DEFAULT=32

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none", "--server.runOnSave=false", "--client.toolbarMode=minimal", "--client.showErrorDetails=false", "--server.enableStaticServing=false", "--server.maxUploadSize=10"]
