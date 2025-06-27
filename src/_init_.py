"""
player_mapping_project - A system for cross-camera player re-identification in sports analytics

This package provides:
- Player detection using YOLOv11
- Multi-camera player matching
- Visualization of tracking results
"""

__version__ = "1.0.0"
__all__ = ['PlayerDetector', 'FeatureExtractor', 'PlayerMatcher', 'VideoProcessor']

# Import key classes for direct package access
from .player_detector import PlayerDetector
from .feature_extractor import FeatureExtractor
from .player_matcher import PlayerMatcher
from .video_processor import VideoProcessor

# Initialize package-level logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Package constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
COLOR_PALETTE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

def get_version():
    """Return package version"""
    return __version__

def validate_environment():
    """Check required dependencies are available"""
    try:
        import torch
        import cv2
        import pandas
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

# Initialize important paths
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(package_dir, 'data')
models_dir = os.path.join(data_dir, 'models')

# Create required directories if they don't exist
os.makedirs(models_dir, exist_ok=True)