import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the euclidean distance between two vectors."""
    return float(np.linalg.norm(v1 - v2))

class FaceMatcher:
    """Matcher that keeps an in-memory index of embeddings to compare against."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.identities = [] # List of user objects or dicts
        self.embeddings = [] # List of numpy arrays
        
    def load_from_db(self, db_session):
        """Load embeddings from the database into memory."""
        from database.models import User, Embedding
        import json
        
        self.identities = []
        self.embeddings = []
        
        users = db_session.query(User).all()
        for user in users:
            for emb in user.embeddings:
                try:
                    vec = np.array(json.loads(emb.vector_data), dtype=np.float32)
                    self.identities.append({
                        "id": user.id,
                        "name": user.name
                    })
                    self.embeddings.append(vec)
                except Exception as e:
                    pass

    def match(self, query_embedding: np.ndarray) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Match a query embedding against the loaded index.
        Returns the best matched identity and the similarity score.
        If no match above threshold is found, returns (None, score).
        """
        if len(self.embeddings) == 0:
            return None, 0.0
            
        max_sim = -1.0
        best_match = None
        
        for i, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            if sim > max_sim:
                max_sim = sim
                best_match = self.identities[i]
                
        if max_sim >= self.threshold:
            return best_match, max_sim
            
        return None, max_sim
