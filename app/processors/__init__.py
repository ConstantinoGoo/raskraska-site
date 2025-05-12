"""
Image processing package for the Raskraska application.
Contains modules for both AI-based and traditional image processing approaches.
"""

from .image_processor import ImageType, create_coloring_page, process_image
from .image_analyzer import ImageAnalyzer

__all__ = ['ImageType', 'create_coloring_page', 'process_image', 'ImageAnalyzer'] 