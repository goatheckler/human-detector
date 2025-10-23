#!/bin/bash

if [ ! -d "src/backend/.venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

. src/backend/.venv/bin/activate

uvicorn src.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
