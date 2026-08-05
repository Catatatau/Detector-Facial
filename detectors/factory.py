from typing import Optional
from config.settings import config
from .base import BaseFaceDetector
from .opencv_haar import OpenCVHaarDetector
from .yunet import YuNetDetector
from utils.logger import logger

def get_detector(backend_name: Optional[str] = None) -> BaseFaceDetector:
    """Factory to get the face detector based on configuration."""
    backend = backend_name or config.models.detection_backend.lower()
    
    logger.info(f"Loading face detector backend: {backend}")
    
    if backend == "yunet":
        return YuNetDetector()
    elif backend == "opencv" or backend == "haar":
        return OpenCVHaarDetector()
    else:
        logger.warning(f"Unknown backend {backend}, falling back to OpenCV Haar")
        return OpenCVHaarDetector()
