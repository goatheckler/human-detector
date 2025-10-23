import pytest
from pydantic import ValidationError
from src.backend.models.bounding_box import BoundingBox


def test_bounding_box_valid():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=0.95)
    assert bbox.x1 == 10.0
    assert bbox.y1 == 20.0
    assert bbox.x2 == 100.0
    assert bbox.y2 == 200.0
    assert bbox.confidence == 0.95


def test_bounding_box_confidence_validation():
    with pytest.raises(ValidationError):
        BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=1.5)
    
    with pytest.raises(ValidationError):
        BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0, confidence=-0.1)
