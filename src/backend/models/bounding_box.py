from pydantic import Field
from src.backend.models.api_model import APIModel


class BoundingBox(APIModel):
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
