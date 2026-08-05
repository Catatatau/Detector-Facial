from typing import Optional
from config.settings import config
from .base import BaseEmbedder
from .insightface_embedder import InsightFaceEmbedder
from utils.logger import logger

def get_embedder(backend_name: Optional[str] = None) -> BaseEmbedder:
    """Factory to get the face embedder based on configuration."""
    backend = backend_name or config.models.recognition_backend.lower()
    
    logger.info(f"Loading face embedder backend: {backend}")
    
    if backend == "insightface":
        return InsightFaceEmbedder()
    else:
        logger.warning(f"Unknown backend {backend}, falling back to InsightFace")
        return InsightFaceEmbedder()
