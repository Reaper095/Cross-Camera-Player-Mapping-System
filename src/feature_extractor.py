import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from pyexpat import features
import pandas as pd
import os

class FeatureExtractor:
    def __init__(self, config: dict = None):
        """
        Initialize feature extractor with optional configuration
        
        Args:
            config: Dictionary containing configuration parameters:
                   - color_hist_bins: Number of bins for color histograms
                   - hog_orientations: Number of HOG orientations
                   - hog_pixels_per_cell: HOG pixels per cell (tuple)
                   - hog_cells_per_block: HOG cells per block (tuple)
        """
        self.config = config or {
            'color_hist_bins': 32,
            'hog_orientations': 9,
            'hog_pixels_per_cell': (8, 8),
            'hog_cells_per_block': (2, 2)
        }
        self.scaler = StandardScaler()
        
    def extract_features(self, crop: np.ndarray, bbox: List[int], video_info: Dict = None) -> Dict:
        """
        Extract all features for a player detection
        
        Args:
            crop: Player image crop
            bbox: Bounding box coordinates [x1,y1,x2,y2]
            video_info: Video metadata dictionary
            
        Returns:
            Dictionary containing all extracted features
        """
        return {
            'visual': self._extract_visual_features(crop, bbox),
            'spatial': self._extract_spatial_features(bbox, video_info)
        }
        
    def _extract_visual_features(self, crop, bbox: List[int]) -> List[float]:
        """Handle both numpy array and list inputs"""
        # Convert to numpy array if needed
        if isinstance(crop, list):
            crop = np.array(crop, dtype=np.uint8)
        
        # Validate input
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            return np.zeros(128 + 32).tolist()
        
        try:
            # Resize maintaining aspect ratio
            h, w = crop.shape[:2]
            target_h = 128
            target_w = int(target_h * (w/h))
            resized = cv2.resize(crop, (target_w, target_h))
            
            # Feature extraction
            color = self._extract_color_histogram(resized)
            hog = self._extract_hog_features(resized)
            return np.concatenate([color, hog]).tolist()
        
        except Exception as e:
            print(f"Feature extraction warning: {str(e)}")
            return np.zeros(128 + 32).tolist()
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        bins = self.config.get('color_hist_bins', 32)
        # ... rest of the method ...
        """
        Extract color histogram features
        
        Args:
            image: Input image region
            bins: Number of histogram bins
            
        Returns:
            Color histogram features
        """
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)
        v_hist = v_hist.flatten() / np.sum(v_hist)
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Ensure valid HOG parameters and handle errors"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate valid dimensions
            height, width = gray.shape
            cell_size = 8  # Fixed cell size
            block_size = 16  # Fixed block size
            
            # Ensure image dimensions are multiples of cell size
            width = (width // cell_size) * cell_size
            height = (height // cell_size) * cell_size
            if width == 0 or height == 0:
                return np.zeros(32)
                
            gray = cv2.resize(gray, (width, height))
            
            # HOG parameters
            win_size = (width, height)
            cell_size = (cell_size, cell_size)
            block_size = (block_size, block_size)
            nbins = 9
            
            hog = cv2.HOGDescriptor(
                _winSize=win_size,
                _blockSize=block_size,
                _blockStride=cell_size,
                _cellSize=cell_size,
                _nbins=nbins
            )
            
            features = hog.compute(gray)
            return features.flatten()[:32]  # Return first 32 features
            
        except Exception as e:
            self.logger.warning(f"HOG extraction failed: {str(e)}")
            return np.zeros(32)
    
    def _extract_spatial_features(self, bbox: List[int], video_info: Dict) -> Dict:
        """Extract spatial features"""
        if video_info is None:
            return {}
            
        return {
            'relative_position': [
                (bbox[0] + bbox[2]) / (2 * video_info['width']),
                (bbox[1] + bbox[3]) / (2 * video_info['height'])
            ],
            'size_ratio': (bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(video_info['width']*video_info['height'])
        }
    
    def extract_temporal_features(self, detections_df: pd.DataFrame,
                                window_size: int = 10) -> Dict:
        """
        Extract temporal features for player matching
        
        Args:
            detections_df: DataFrame with detection data
            window_size: Number of frames to consider
            
        Returns:
            Dictionary with temporal features
        """
        temporal_features = {}
        
        for video_source in detections_df['video_source'].unique():
            video_data = detections_df[detections_df['video_source'] == video_source]
            
            # Group detections by frame
            frame_groups = video_data.groupby('frame_idx')
            
            features_by_frame = []
            
            for frame_idx, frame_data in frame_groups:
                frame_features = []
                
                for _, detection in frame_data.iterrows():
                    # Get detection features for this frame
                    det_features = {
                        'position': [detection['center_x'], detection['center_y']],
                        'velocity': [detection['velocity_x'], detection['velocity_y']],
                        'size': detection['area'],
                        'confidence': detection['confidence']
                    }
                    frame_features.append(det_features)
                
                features_by_frame.append({
                    'frame_idx': frame_idx,
                    'detections': frame_features
                })
            
            temporal_features[video_source] = features_by_frame
        
        return temporal_features
    
    def compute_feature_similarity(self, features1: np.ndarray, 
                                 features2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if len(features1) != len(features2):
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return (similarity + 1) / 2  # Normalize to 0-1 range
    
    def build_feature_matrix(self, video_path: str, detections_df: pd.DataFrame,
                           max_frames: int = 100) -> np.ndarray:
        """
        Build feature matrix for all detections in a video
        
        Args:
            video_path: Path to video file
            detections_df: DataFrame with detection data
            max_frames: Maximum number of frames to process
            
        Returns:
            Feature matrix (n_detections, n_features)
        """
        cap = cv2.VideoCapture(video_path)
        
        feature_matrix = []
        processed_frames = 0
        
        for frame_idx in sorted(detections_df['frame_idx'].unique()):
            if processed_frames >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Get detections for this frame
            frame_detections = detections_df[detections_df['frame_idx'] == frame_idx]
            
            for _, detection in frame_detections.iterrows():
                # Extract visual features
                visual_features = self.extract_visual_features(frame, detection['bbox'])
                
                # Extract spatial features
                spatial_features = np.array([
                    detection['rel_x'], detection['rel_y'],
                    detection['velocity_x'], detection['velocity_y'],
                    detection['speed'], detection['area']
                ])
                
                # Combine features
                combined_features = np.concatenate([visual_features, spatial_features])
                feature_matrix.append(combined_features)
            
            processed_frames += 1
        
        cap.release()
        
        return np.array(feature_matrix)