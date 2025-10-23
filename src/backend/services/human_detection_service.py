from ultralytics import YOLO
import numpy as np
import base64
import cv2
import torch
from typing import List, Dict, Optional
from src.backend.models.bounding_box import BoundingBox
from src.backend.models.detection_response import DetectionResponse
from src.backend.models.yolo_model_size import YoloModelSize
from src.backend.models.device_type import DeviceType
from src.backend.config import settings


class HumanDetectionService:
    def __init__(
        self, 
        model_size: YoloModelSize = YoloModelSize.NANO, 
        confidence_threshold: float = 0.5,
        supported_devices: List[DeviceType] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0
        self.supported_devices = supported_devices or [DeviceType.CPU, DeviceType.GPU]
        self.models: Dict[DeviceType, Optional[YOLO]] = {}
        
        for device in self.supported_devices:
            try:
                if device == DeviceType.GPU and not torch.cuda.is_available():
                    self.models[device] = None
                else:
                    model = YOLO(model_size.value)
                    model.to(device.value)
                    self.models[device] = model
            except Exception:
                self.models[device] = None
    
    def _decode_image(self, base64_image: str) -> np.ndarray:
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    
    def detect_humans(
        self, 
        base64_image: str, 
        device: DeviceType = DeviceType.GPU,
        cpu_threads: Optional[int] = None
    ) -> DetectionResponse:
        if device not in self.supported_devices:
            available = [d.value for d in self.supported_devices if self.models.get(d)]
            raise ValueError(
                f"Device '{device.value}' is not supported. "
                f"Available devices: {available}"
            )
        
        model = self.models.get(device)
        if model is None:
            available = [d.value for d in self.supported_devices if self.models.get(d)]
            raise ValueError(
                f"Device '{device.value}' requested but not available in this environment. "
                f"Available devices: {available}"
            )
        
        if device == DeviceType.CPU and cpu_threads is not None:
            original_threads = torch.get_num_threads()
            torch.set_num_threads(cpu_threads)
        
        image = self._decode_image(base64_image)
        
        results = model(
            image,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            verbose=False
        )
        
        if device == DeviceType.CPU and cpu_threads is not None:
            torch.set_num_threads(original_threads)
        
        bounding_boxes: List[BoundingBox] = []
        max_confidence = 0.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bounding_boxes.append(
                    BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence
                    )
                )
                max_confidence = max(max_confidence, confidence)
        
        human_detected = len(bounding_boxes) > 0
        
        return DetectionResponse(
            human_detected=human_detected,
            bounding_boxes=bounding_boxes,
            max_confidence=max_confidence if human_detected else 0.0
        )
