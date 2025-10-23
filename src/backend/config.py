from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from src.backend.models.yolo_model_size import YoloModelSize
from src.backend.models.device_type import DeviceType
from typing import List, Optional
import torch
import os


CPU_THREADS_MIN = 1
CPU_THREADS_MAX = 64
CPU_THREADS_DEFAULT = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUMAN_DETECTOR_",
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    model_size: YoloModelSize = YoloModelSize.NANO
    confidence_threshold: float = 0.5
    supported_devices: List[DeviceType] = [DeviceType.CPU, DeviceType.GPU]
    cpu_threads: int = CPU_THREADS_DEFAULT
    
    @field_validator('cpu_threads')
    @classmethod
    def validate_cpu_threads(cls, v: int) -> int:
        if v < CPU_THREADS_MIN or v > CPU_THREADS_MAX:
            raise ValueError(
                f"cpu_threads must be between {CPU_THREADS_MIN} and {CPU_THREADS_MAX}, got {v}"
            )
        return v


settings = Settings()

torch.set_num_threads(settings.cpu_threads)
os.environ['OMP_NUM_THREADS'] = str(settings.cpu_threads)
os.environ['MKL_NUM_THREADS'] = str(settings.cpu_threads)
