import cv2
import numpy as np
from typing import List, Dict, Any
from .base import BaseFaceDetector
from utils.logger import logger

class OpenCVHaarDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascades."""
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: tuple = (30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load the pre-trained Haar cascade classifier for faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.classifier = cv2.CascadeClassifier(cascade_path)
        
        if self.classifier.empty():
            logger.error(f"Failed to load Haar cascade from {cascade_path}")
            
        logger.info("Initialized OpenCV Haar Cascade Detector")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.classifier.empty():
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.classifier.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors, 
            minSize=self.min_size
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": 1.0, # Haar cascades don't provide confidence easily
                "landmarks": None
            })
            
        return results
