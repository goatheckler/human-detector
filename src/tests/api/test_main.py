import pytest
from fastapi.testclient import TestClient
import base64
import numpy as np
import cv2
from src.backend.api.main import app


def create_test_image() -> str:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_detect_endpoint_valid_image():
    test_image = create_test_image()
    response = client.post("/detect", json={"image_data": test_image, "device": "cpu"})
    
    assert response.status_code == 200
    data = response.json()
    assert "humanDetected" in data
    assert "boundingBoxes" in data
    assert "maxConfidence" in data
    assert isinstance(data["humanDetected"], bool)
    assert isinstance(data["boundingBoxes"], list)
    assert isinstance(data["maxConfidence"], float)


def test_detect_endpoint_invalid_base64():
    response = client.post("/detect", json={"image_data": "invalid!@#$", "device": "cpu"})
    assert response.status_code == 422


def test_detect_endpoint_explicit_gpu():
    test_image = create_test_image()
    response = client.post("/detect", json={"image_data": test_image, "device": "cuda"})
    assert response.status_code in [200, 400]
