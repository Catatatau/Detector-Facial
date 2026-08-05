import cv2
import numpy as np
from typing import List, Tuple

# Standard ArcFace reference points for 112x112 image
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(image: np.ndarray, landmarks: List[List[int]], output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Align the face based on 5 landmarks (left eye, right eye, nose, left mouth, right mouth).
    Returns an aligned crop of output_size.
    """
    if landmarks is None or len(landmarks) < 5:
        return None
        
    # Convert landmarks to numpy array
    src_pts = np.array(landmarks, dtype=np.float32)
    
    # Calculate similarity transform
    # cv2.estimateAffinePartial2D finds the optimal translation, rotation, and uniform scale
    tform, inliers = cv2.estimateAffinePartial2D(src_pts, REFERENCE_FACIAL_POINTS)
    
    if tform is None:
        return None
        
    # Apply the transform
    aligned_face = cv2.warpAffine(image, tform, output_size, borderValue=0.0)
    
    return aligned_face

def crop_face(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Fallback to simple crop if no landmarks are available."""
    x, y, w, h = bbox
    # Ensure within bounds
    x = max(0, x)
    y = max(0, y)
    h_img, w_img, _ = image.shape
    x_end = min(w_img, x + w)
    y_end = min(h_img, y + h)
    
    crop = image[y:y_end, x:x_end]
    return crop
