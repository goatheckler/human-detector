from pydantic import Field, field_validator
from src.backend.models.api_model import APIModel
from src.backend.models.device_type import DeviceType
from src.backend.config import CPU_THREADS_MIN, CPU_THREADS_MAX, CPU_THREADS_DEFAULT
from typing import Optional
import base64


class DetectionRequest(APIModel):
    image_data: str = Field(
        ..., 
        description="Base64-encoded image data. Supported formats: JPEG, PNG, GIF, WebP, BMP, TIFF"
    )
    device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Device to use for inference: 'cpu' or 'cuda' (GPU)"
    )
    cpu_threads: Optional[int] = Field(
        default=None,
        description=f"Number of CPU threads for inference (only applies to CPU device). Range: {CPU_THREADS_MIN}-{CPU_THREADS_MAX}. Defaults to server setting if not specified."
    )
    
    @field_validator('image_data')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        return v
    
    @field_validator('cpu_threads')
    @classmethod
    def validate_cpu_threads(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < CPU_THREADS_MIN or v > CPU_THREADS_MAX):
            raise ValueError(
                f"cpu_threads must be between {CPU_THREADS_MIN} and {CPU_THREADS_MAX}, got {v}"
            )
        return v
