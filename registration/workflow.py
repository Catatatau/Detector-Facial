import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from database.session import SessionLocal
from database.models import User, Embedding
from preprocessing.quality import check_image_quality
from preprocessing.alignment import align_face, crop_face
from detectors.factory import get_detector
from embeddings.factory import get_embedder
from utils.logger import logger

class RegistrationWorkflow:
    def __init__(self, required_samples: int = 5):
        self.required_samples = required_samples
        self.collected_samples = []
        self.detector = get_detector()
        self.embedder = get_embedder()
        self.current_user_name = None
        
    def start_registration(self, name: str):
        """Begin collecting samples for a new user."""
        self.current_user_name = name
        self.collected_samples = []
        logger.info(f"Started registration for user: {name}")
        
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, str, int]:
        """
        Process a single frame for registration.
        Returns (success, message, current_sample_count).
        """
        if self.current_user_name is None:
            return False, "Registration not started", 0
            
        if len(self.collected_samples) >= self.required_samples:
            return True, "All samples collected", len(self.collected_samples)

        # Detect face
        faces = self.detector.detect(frame)
        if not faces:
            return False, "No face detected", len(self.collected_samples)
        if len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only one face is visible.", len(self.collected_samples)
            
        face = faces[0]
        bbox = face["bbox"]
        landmarks = face.get("landmarks")
        
        # Check quality before heavy processing
        crop = crop_face(frame, bbox)
        is_good, msg = check_image_quality(crop)
        if not is_good:
            return False, msg, len(self.collected_samples)
            
        # Align
        if landmarks:
            aligned = align_face(frame, landmarks)
            if aligned is None:
                aligned = cv2.resize(crop, (112, 112))
        else:
            aligned = cv2.resize(crop, (112, 112))
            
        # Get embedding
        embedding = self.embedder.get_embedding(aligned)
        if embedding is None:
            return False, "Failed to extract embedding", len(self.collected_samples)
            
        self.collected_samples.append(embedding)
        return True, "Sample collected", len(self.collected_samples)

    def complete_registration(self) -> bool:
        """Save the collected samples to the database."""
        if len(self.collected_samples) < self.required_samples:
            logger.warning("Not enough samples to complete registration.")
            return False
            
        db = SessionLocal()
        try:
            # Check if user exists
            user = db.query(User).filter(User.name == self.current_user_name).first()
            if not user:
                user = User(name=self.current_user_name)
                db.add(user)
                db.flush() # to get user.id
                
            for emb in self.collected_samples:
                # Store embedding as JSON string
                db_emb = Embedding(
                    user_id=user.id,
                    vector_data=json.dumps(emb.tolist())
                )
                db.add(db_emb)
                
            db.commit()
            logger.info(f"Successfully registered user {self.current_user_name}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to register user: {e}")
            return False
        finally:
            db.close()
            self.current_user_name = None
            self.collected_samples = []
