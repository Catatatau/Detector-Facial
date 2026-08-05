from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class BaseFaceDetector(ABC):
    """Abstract base class for face detectors."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the given image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            A list of dictionaries, where each dict represents a detected face:
            {
                "bbox": [x, y, w, h],
                "confidence": float,
                "landmarks": [[x1, y1], [x2, y2], ...] (optional)
            }
        """
        pass
