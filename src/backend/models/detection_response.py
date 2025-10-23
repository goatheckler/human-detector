from pydantic import Field
from typing import List
from src.backend.models.api_model import APIModel
from src.backend.models.bounding_box import BoundingBox


class DetectionResponse(APIModel):
    human_detected: bool = Field(..., description="Whether a human was detected in the image")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="List of detected human bounding boxes")
    max_confidence: float = Field(..., ge=0.0, le=1.0, description="Maximum confidence score across all detected humans")
