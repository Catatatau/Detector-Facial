import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import Optional
from .base import BaseEmbedder
from utils.logger import logger

class InsightFaceEmbedder(BaseEmbedder):
    """Face embedder using InsightFace (ArcFace)."""
    
    def __init__(self, name: str = "buffalo_l", ctx_id: int = 0):
        # We only need recognition ('rec') since detection is handled separately if we want to decouple
        # But FaceAnalysis requires detection to work fully if we pass full images.
        # However, we can use the recognition model directly if we pass the aligned face crop.
        # The easiest and most robust way using the insightface package is to load the model.
        try:
            # We must load detection as well because FaceAnalysis asserts its presence,
            # even though we only extract the recognition model.
            self.app = FaceAnalysis(name=name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            # Extract just the recognition model to bypass the built-in detection if we want
            self.rec_model = self.app.models['recognition']
            logger.info(f"Initialized InsightFace Embedder ({name})")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            self.rec_model = None

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding. The face_image must be a 112x112 aligned BGR image 
        for optimal results with arcface models.
        """
        if self.rec_model is None or face_image is None or face_image.size == 0:
            return None
            
        # The recognition model in InsightFace has a method `get_feat` which takes an aligned face.
        # But typically we can just pass the image.
        # The exact method signature varies by InsightFace version.
        try:
            # For typical arcface models in insightface:
            # We can use get_feat if available
            if hasattr(self.rec_model, 'get_feat'):
                embedding = self.rec_model.get_feat(face_image)
            else:
                # If not, we might need to construct a dummy Face object, but get_feat is standard for ArcFaceONNX
                embedding = self.rec_model.get_feat(face_image)
                
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
