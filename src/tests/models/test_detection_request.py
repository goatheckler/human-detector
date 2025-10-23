import pytest
import base64
from pydantic import ValidationError
from src.backend.models.detection_request import DetectionRequest


def test_detection_request_valid():
    valid_image = base64.b64encode(b"fake image data").decode('utf-8')
    request = DetectionRequest(image_data=valid_image)
    assert request.image_data == valid_image


def test_detection_request_invalid_base64():
    with pytest.raises(ValidationError):
        DetectionRequest(image_data="not valid base64!@#$")


def test_detection_request_empty_string():
    request = DetectionRequest(image_data="")
    assert request.image_data == ""
