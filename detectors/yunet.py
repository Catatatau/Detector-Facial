import cv2
import numpy as np
import os
from typing import List, Dict, Any
from .base import BaseFaceDetector
from utils.logger import logger
from utils.downloader import download_file

# YuNet model URL from OpenCV model zoo
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

class YuNetDetector(BaseFaceDetector):
    """Face detector using OpenCV YuNet."""
    
    def __init__(self, model_path: str = "models/face_detection_yunet_2023mar.onnx", conf_threshold: float = 0.6, nms_threshold: float = 0.3, top_k: int = 5000):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        
        # Ensure model exists
        if not os.path.exists(self.model_path):
            logger.info("YuNet model not found locally.")
            success = download_file(YUNET_URL, self.model_path)
            if not success:
                logger.error("Could not download YuNet model.")
                return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
            logger.info("Initialized YuNet Detector")
        except Exception as e:
            logger.error(f"Failed to initialize YuNet: {e}")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.detector is None:
            return []
            
        height, width, _ = image.shape
        # YuNet requires setting the input size to match the image or resizing the image
        self.detector.setInputSize((width, height))
        
        _, faces = self.detector.detect(image)
        
        results = []
        if faces is not None:
            for face in faces:
                # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
                score = float(face[14])
                
                # 5 landmarks: right eye, left eye, nose tip, right corner of mouth, left corner of mouth
                landmarks = [
                    [int(face[4]), int(face[5])],
                    [int(face[6]), int(face[7])],
                    [int(face[8]), int(face[9])],
                    [int(face[10]), int(face[11])],
                    [int(face[12]), int(face[13])]
                ]
                
                results.append({
                    "bbox": bbox,
                    "confidence": score,
                    "landmarks": landmarks
                })
                
        return results
