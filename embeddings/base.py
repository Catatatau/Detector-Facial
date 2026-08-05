from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """Abstract base class for face embeddings."""
    
    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate an embedding vector from an aligned face image.
        
        Args:
            face_image: Aligned BGR face image.
            
        Returns:
            1D numpy array representing the embedding.
        """
        pass
