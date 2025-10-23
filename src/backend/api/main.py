from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from src.backend.models.detection_request import DetectionRequest
from src.backend.models.detection_response import DetectionResponse
from src.backend.models.device_type import DeviceType
from src.backend.services.human_detection_service import HumanDetectionService
from src.backend.config import settings
import base64
from typing import Optional

app = FastAPI(
    title="Human Detection API", 
    version="1.0.0",
    response_model_by_alias=True
)

detection_service = HumanDetectionService(
    model_size=settings.model_size,
    confidence_threshold=settings.confidence_threshold,
    supported_devices=settings.supported_devices
)


@app.post("/detect", response_model=DetectionResponse)
async def detect_humans(
    image: UploadFile = File(...),
    device: str = Form("cpu"),
    cpu_threads: Optional[int] = Form(None)
) -> DetectionResponse:
    """
    Detect humans in an image using YOLO11.
    
    - **image**: Image file (JPEG, PNG, GIF, WebP, BMP, TIFF)
    - **device**: Device to use for inference ('cpu' or 'gpu'). Defaults to 'cpu'
    - **cpu_threads**: Number of CPU threads (optional, uses server default if not specified)
    - Returns bounding boxes for all detected humans with confidence scores
    
    Unsupported formats (HEIC, AVIF, RAW) will return a 400 error.
    """
    try:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        device_type = DeviceType.GPU if device.lower() == "gpu" else DeviceType.CPU
        
        return detection_service.detect_humans(
            image_base64,
            device_type,
            cpu_threads
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
