import pytest
import base64
from pathlib import Path
from src.backend.services.human_detection_service import HumanDetectionService
from src.backend.models.device_type import DeviceType


@pytest.fixture
def detection_service():
    return HumanDetectionService(supported_devices=[DeviceType.CPU])


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent.parent / "fixtures" / "images"


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_images_without_humans_return_false(detection_service, fixtures_dir):
    without_humans_dir = fixtures_dir / "without_humans"
    image_files = list(without_humans_dir.glob("*.jpg"))
    
    assert len(image_files) > 0, "No test images found in without_humans directory"
    
    for image_path in image_files:
        base64_image = image_to_base64(image_path)
        response = detection_service.detect_humans(base64_image, device=DeviceType.CPU)
        
        assert response.human_detected is False, f"Expected no humans in {image_path.name}"
        assert len(response.bounding_boxes) == 0, f"Expected no bounding boxes in {image_path.name}"
        assert response.max_confidence == 0.0, f"Expected 0.0 confidence for {image_path.name}"


def test_placeholder_stick_figure_not_detected(detection_service, fixtures_dir):
    stick_figure = fixtures_dir / "with_humans" / "stick_figure.jpg"
    
    assert stick_figure.exists(), "stick_figure.jpg not found"
    
    base64_image = image_to_base64(stick_figure)
    response = detection_service.detect_humans(base64_image, device=DeviceType.CPU)
    
    assert response.human_detected is False, "Stick figure should not be detected by YOLO (not a real human)"
    assert len(response.bounding_boxes) == 0


def test_fixture_directories_exist(fixtures_dir):
    with_humans = fixtures_dir / "with_humans"
    without_humans = fixtures_dir / "without_humans"
    
    assert with_humans.exists(), "with_humans directory should exist"
    assert without_humans.exists(), "without_humans directory should exist"
    
    without_humans_images = list(without_humans.glob("*.jpg"))
    assert len(without_humans_images) >= 4, f"Expected at least 4 images without humans, found {len(without_humans_images)}"
