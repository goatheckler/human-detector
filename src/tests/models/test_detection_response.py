import pytest
from pydantic import ValidationError
from src.backend.models.detection_response import DetectionResponse
from src.backend.models.bounding_box import BoundingBox


def test_detection_response_with_humans():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.95)
    response = DetectionResponse(
        human_detected=True,
        bounding_boxes=[bbox],
        max_confidence=0.95
    )
    assert response.human_detected is True
    assert len(response.bounding_boxes) == 1
    assert response.max_confidence == 0.95


def test_detection_response_no_humans():
    response = DetectionResponse(
        human_detected=False,
        bounding_boxes=[],
        max_confidence=0.0
    )
    assert response.human_detected is False
    assert len(response.bounding_boxes) == 0
    assert response.max_confidence == 0.0


def test_detection_response_confidence_validation():
    with pytest.raises(ValidationError):
        DetectionResponse(human_detected=True, bounding_boxes=[], max_confidence=1.5)
