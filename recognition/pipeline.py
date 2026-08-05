import cv2
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from database.session import SessionLocal
from database.models import RecognitionEvent
from detectors.factory import get_detector
from embeddings.factory import get_embedder
from matching.matcher import FaceMatcher
from tracking.tracker import CentroidTracker
from preprocessing.alignment import align_face, crop_face
from preprocessing.quality import check_image_quality
from config.settings import config
from utils.logger import logger

class RecognitionPipeline:
    def __init__(self):
        self.detector = get_detector()
        self.embedder = get_embedder()
        self.matcher = FaceMatcher(threshold=config.models.recognition_threshold)
        self.tracker = CentroidTracker()
        
        self.db_session = SessionLocal()
        # Load known embeddings into memory
        self.matcher.load_from_db(self.db_session)
        logger.info(f"Loaded {len(self.matcher.embeddings)} embeddings for recognition.")
        
        # Cache for recognized identities to avoid re-computing every frame
        # Maps tracker object_id -> (identity_dict, score)
        self.identity_cache = {} 
        
    def reload_matcher(self):
        """Reload embeddings from the database."""
        self.matcher.load_from_db(self.db_session)
        self.identity_cache.clear()
        logger.info(f"Reloaded {len(self.matcher.embeddings)} embeddings for recognition.")

    def log_event(self, identity: Dict[str, Any], score: float, latency: float):
        """Log the recognition event to the database."""
        try:
            event = RecognitionEvent(
                user_id=identity["id"] if identity else None,
                confidence_score=score,
                processing_latency=latency,
                camera_id=str(config.camera.index)
            )
            self.db_session.add(event)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to log event: {e}")

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a frame for recognition.
        Returns a list of dicts with bbox, identity, and score.
        """
        start_time = time.time()
        
        faces = self.detector.detect(frame)
        
        # Update tracker
        rects = [f["bbox"] for f in faces]
        objects = self.tracker.update(rects)
        
        # Map objects to bounding boxes
        # The tracker doesn't return the exact bbox, just the centroid. 
        # We need to associate the detected faces with the tracker IDs.
        results = []
        
        # Simple association: match centroids
        for object_id, centroid in objects.items():
            best_face = None
            min_dist = float('inf')
            
            for face in faces:
                x, y, w, h = face["bbox"]
                cx, cy = x + w/2, y + h/2
                dist = (cx - centroid[0])**2 + (cy - centroid[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_face = face
                    
            if best_face is None or min_dist > 2500: # 50 pixels distance squared threshold
                continue
                
            # Now we have the tracked object and its current bounding box
            # If we don't have this object in cache, or occasionally, we run recognition
            if object_id not in self.identity_cache:
                identity, score = self._recognize_face(frame, best_face)
                self.identity_cache[object_id] = (identity, score)
                
                # Log the event when a new face is recognized
                latency = (time.time() - start_time) * 1000
                self.log_event(identity, score, latency)
            else:
                identity, score = self.identity_cache[object_id]
                
            results.append({
                "bbox": best_face["bbox"],
                "landmarks": best_face.get("landmarks"),
                "identity": identity,
                "score": score,
                "tracker_id": object_id
            })
            
        return results

    def _recognize_face(self, frame: np.ndarray, face: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        """Extract embedding and match against database."""
        bbox = face["bbox"]
        landmarks = face.get("landmarks")
        
        crop = crop_face(frame, bbox)
        if crop is None or crop.size == 0:
            return None, 0.0
            
        # Optional: check quality
        is_good, _ = check_image_quality(crop)
        if not is_good:
            # Maybe return a specific "Poor Quality" identity or just unknown
            return None, 0.0
            
        if landmarks:
            aligned = align_face(frame, landmarks)
            if aligned is None:
                aligned = cv2.resize(crop, (112, 112))
        else:
            aligned = cv2.resize(crop, (112, 112))
            
        embedding = self.embedder.get_embedding(aligned)
        if embedding is None:
            return None, 0.0
            
        return self.matcher.match(embedding)
        
    def __del__(self):
        if hasattr(self, 'db_session'):
            self.db_session.close()
