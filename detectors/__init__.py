"""
Detectors module.
"""
from .base import BaseFaceDetector
from .factory import get_detector

__all__ = ['BaseFaceDetector', 'get_detector']
