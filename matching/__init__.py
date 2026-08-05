"""
Matching module.
"""
from .matcher import FaceMatcher, cosine_similarity, euclidean_distance

__all__ = ['FaceMatcher', 'cosine_similarity', 'euclidean_distance']
