import cv2
import threading
import time
from typing import Optional, Tuple
from utils.logger import logger

class CameraStream:
    """Threaded camera stream to avoid blocking the main thread."""
    def __init__(self, src: int = 0, resolution: Tuple[int, int] = (1280, 720), fps: int = 30):
        self.src = src
        self.resolution = resolution
        self.target_fps = fps
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._lock = threading.Lock()
        
        if not self.grabbed:
            logger.error(f"Failed to open camera source {self.src}")
        else:
            logger.info(f"Camera {self.src} started at {self.resolution[0]}x{self.resolution[1]}")

    def start(self) -> "CameraStream":
        """Start the thread to read frames from the video stream."""
        if self.grabbed:
            t = threading.Thread(target=self.update, args=(), daemon=True)
            t.start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while not self.stopped:
            # If the stream is closed, stop the thread
            if not self.stream.isOpened():
                self.stop()
                break
                
            grabbed, frame = self.stream.read()
            
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # Sleep slightly to prevent high CPU usage if camera is fast
            time.sleep(1.0 / (self.target_fps * 2))

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Return the most recent frame."""
        with self._lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        """Indicate that the thread should be stopped."""
        logger.info(f"Stopping camera {self.src}")
        self.stopped = True
        time.sleep(0.1) # wait for thread to finish
        if self.stream.isOpened():
            self.stream.release()

    def __enter__(self):
        return self.start()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
