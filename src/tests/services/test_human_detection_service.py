import pytest
import base64
import numpy as np
import cv2
from src.backend.services.human_detection_service import HumanDetectionService
from src.backend.models.device_type import DeviceType


def create_test_image() -> str:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def test_human_detection_service_initialization():
    service = HumanDetectionService()
    assert service.models is not None
    assert service.confidence_threshold == 0.5
    assert service.person_class_id == 0


def test_detect_humans_no_humans():
    service = HumanDetectionService(supported_devices=[DeviceType.CPU])
    test_image = create_test_image()
    
    response = service.detect_humans(test_image, device=DeviceType.CPU)
    
    assert response is not None
    assert isinstance(response.human_detected, bool)
    assert isinstance(response.bounding_boxes, list)
    assert isinstance(response.max_confidence, float)


def test_invalid_base64():
    service = HumanDetectionService(supported_devices=[DeviceType.CPU])
    
    with pytest.raises(Exception):
        service.detect_humans("invalid base64", device=DeviceType.CPU)


def test_unsupported_device():
    service = HumanDetectionService(supported_devices=[DeviceType.CPU])
    test_image = create_test_image()
    
    with pytest.raises(ValueError, match="not supported"):
        service.detect_humans(test_image, device=DeviceType.GPU)
