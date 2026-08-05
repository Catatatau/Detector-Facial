import cv2
import numpy as np

def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if an image is blurry using the variance of the Laplacian.
    Args:
        image: BGR or Grayscale image.
        threshold: Variance threshold. Lower means it allows more blur.
    Returns:
        True if blurry, False otherwise.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def check_lighting(image: np.ndarray, min_brightness: int = 40, max_brightness: int = 220) -> bool:
    """
    Check if the image is too dark or too bright.
    Returns True if lighting is good.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    mean_brightness = np.mean(gray)
    return min_brightness <= mean_brightness <= max_brightness

def check_image_quality(image: np.ndarray) -> tuple[bool, str]:
    """
    Comprehensive quality check for registration.
    Returns (is_good, reason).
    """
    if image is None or image.size == 0:
        return False, "Empty image"
        
    if is_blurry(image):
        return False, "Image is too blurry"
        
    if not check_lighting(image):
        return False, "Poor lighting conditions"
        
    return True, "Good"
