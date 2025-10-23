#!/bin/bash

if [ ! -d "src/frontend/.venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

. src/frontend/.venv/bin/activate

export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"

echo "Starting Streamlit UI..."
echo "API endpoint: $API_BASE_URL"
echo ""

streamlit run src/frontend/app.py --server.port 8501
