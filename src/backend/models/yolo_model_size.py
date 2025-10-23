from enum import Enum


class YoloModelSize(str, Enum):
    NANO = "yolo11n.pt"
    SMALL = "yolo11s.pt"
    MEDIUM = "yolo11m.pt"
    LARGE = "yolo11l.pt"
    XLARGE = "yolo11x.pt"
