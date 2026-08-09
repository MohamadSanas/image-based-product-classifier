"""
core — Pure domain logic, framework-agnostic data classes and enums.
No dependency on OpenCV, TensorFlow, FastAPI, etc.
"""
from src.core.entities import BoundingBox, DetectedProduct, ScanResult
from src.core.interfaces import IClassifier, IDetector, IPreprocessor, IScanRepository

__all__ = [
    "BoundingBox",
    "DetectedProduct",
    "ScanResult",
    "IClassifier",
    "IDetector",
    "IPreprocessor",
    "IScanRepository",
]
