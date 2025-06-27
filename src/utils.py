import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('player_mapping.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_directories(base_path: str = "output") -> Dict[str, str]:
    """
    Create necessary output directories for the project.
    
    Args:
        base_path: Base output directory path
    
    Returns:
        Dictionary with paths to created directories
    """
    directories = {
        'base': base_path,
        'detections': os.path.join(base_path, 'detections'),
        'visualizations': os.path.join(base_path, 'visualizations'),
        'results': os.path.join(base_path, 'results')
    }
    
    for dir_path in directories.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return directories

def validate_file_paths(file_paths: Dict[str, str]) -> bool:
    """
    Validate that all required files exist.
    
    Args:
        file_paths: Dictionary of file descriptions and their paths
    
    Returns:
        True if all files exist, False otherwise
    """
    missing_files = []
    
    for description, path in file_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{description}: {path}")
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def load_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract basic information from a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection and union areas
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def normalize_coordinates(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
    
    Returns:
        Normalized bounding box coordinates
    """
    return [
        bbox[0] / img_width,
        bbox[1] / img_height,
        bbox[2] / img_width,
        bbox[3] / img_height
    ]

def denormalize_coordinates(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Denormalize bounding box coordinates from [0, 1] range to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
    
    Returns:
        Pixel coordinates as integers
    """
    return [
        int(bbox[0] * img_width),
        int(bbox[1] * img_height),
        int(bbox[2] * img_width),
        int(bbox[3] * img_height)
    ]

def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Center point (x, y)
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def calculate_bbox_area(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Area of the bounding box
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def save_detection_results(detections: Dict[str, Any], output_path: str) -> None:
    """
    Save detection results to a JSON file.
    
    Args:
        detections: Detection results dictionary
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2, cls=NumpyEncoder)

def load_detection_results(input_path: str) -> Dict[str, Any]:
    """
    Load detection results from a JSON file.
    
    Args:
        input_path: Path to the JSON file
    
    Returns:
        Detection results dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)

def save_mappings_to_csv(mappings: Dict[int, Dict[str, Any]], output_path: str, 
                        logger: Optional[logging.Logger] = None) -> None:
    """
    Save player mappings to a CSV file, handling both old and new mapping formats.
    
    Args:
        mappings: Player mapping dictionary
        output_path: Path to save the CSV file
        logger: Optional logger instance
    """
    rows = []
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for frame_id, frame_data in mappings.items():
        # Handle both old and new mapping formats
        frame_mappings = frame_data.get('mappings', {})
        timestamp = frame_data.get('timestamp', 0.0)
        tacticam_timestamp = frame_data.get('tacticam_timestamp', 0.0)
        time_difference = frame_data.get('time_difference', 0.0)
        
        for broadcast_id, match_data in frame_mappings.items():
            if isinstance(match_data, dict):
                # New enhanced format with confidence
                tacticam_id = match_data.get('tacticam_id', match_data.get('id'))
                confidence = match_data.get('confidence', 0.0)
                distance = match_data.get('distance', 0.0)
                spatial_confidence = match_data.get('spatial_confidence', 0.0)
            else:
                # Old format (just the tacticam ID)
                tacticam_id = match_data
                confidence = 0.0
                distance = 0.0
                spatial_confidence = 0.0
            
            rows.append({
                'frame_id': frame_id,
                'broadcast_player_id': broadcast_id,
                'tacticam_player_id': tacticam_id,
                'confidence': confidence,
                'distance': distance,
                'spatial_confidence': spatial_confidence,
                'timestamp': timestamp,
                'tacticam_timestamp': tacticam_timestamp,
                'time_difference': time_difference
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Mappings saved to: {output_path}")
        
        # Log confidence statistics
        if 'confidence' in df.columns and not df['confidence'].empty:
            logger.info(f"Confidence statistics - Mean: {df['confidence'].mean():.3f}, "
                       f"Min: {df['confidence'].min():.3f}, Max: {df['confidence'].max():.3f}")
    else:
        logger.warning("No mappings to save")

def save_enhanced_mappings_to_csv(mappings: Dict[int, Dict[str, Any]], output_path: str,
                                logger: Optional[logging.Logger] = None) -> None:
    """
    Save enhanced player mappings to CSV with detailed confidence metrics.
    This is an alias for save_mappings_to_csv for backward compatibility.
    
    Args:
        mappings: Player mapping dictionary
        output_path: Path to save the CSV file
        logger: Optional logger instance
    """
    save_mappings_to_csv(mappings, output_path, logger)

def load_mappings_from_csv(input_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load player mappings from a CSV file and convert back to mapping format.
    
    Args:
        input_path: Path to the CSV file
    
    Returns:
        Player mapping dictionary
    """
    df = pd.read_csv(input_path)
    mappings = {}
    
    for _, row in df.iterrows():
        frame_id = int(row['frame_id'])
        
        if frame_id not in mappings:
            mappings[frame_id] = {
                'mappings': {},
                'timestamp': row.get('timestamp', 0.0),
                'tacticam_timestamp': row.get('tacticam_timestamp', 0.0),
                'time_difference': row.get('time_difference', 0.0)
            }
        
        # Create enhanced mapping data
        mapping_data = {
            'tacticam_id': row['tacticam_player_id'],
            'confidence': row.get('confidence', 0.0),
            'distance': row.get('distance', 0.0),
            'spatial_confidence': row.get('spatial_confidence', 0.0)
        }
        
        mappings[frame_id]['mappings'][row['broadcast_player_id']] = mapping_data
    
    return mappings

def validate_mapping_confidence(mappings: Dict[int, Dict[str, Any]], 
                              min_confidence: float = 0.1) -> Dict[int, Dict[str, Any]]:
    """
    Filter mappings based on confidence threshold.
    
    Args:
        mappings: Player mapping dictionary
        min_confidence: Minimum confidence threshold
    
    Returns:
        Filtered mapping dictionary
    """
    filtered_mappings = {}
    
    for frame_id, frame_data in mappings.items():
        filtered_frame = {
            'mappings': {},
            'timestamp': frame_data.get('timestamp', 0.0),
            'tacticam_timestamp': frame_data.get('tacticam_timestamp', 0.0),
            'time_difference': frame_data.get('time_difference', 0.0)
        }
        
        for broadcast_id, match_data in frame_data.get('mappings', {}).items():
            if isinstance(match_data, dict):
                confidence = match_data.get('confidence', 0.0)
            else:
                confidence = 0.0
            
            if confidence >= min_confidence:
                filtered_frame['mappings'][broadcast_id] = match_data
        
        if filtered_frame['mappings']:  # Only include frames with valid mappings
            filtered_mappings[frame_id] = filtered_frame
    
    return filtered_mappings

def calculate_mapping_statistics(mappings: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for player mappings.
    
    Args:
        mappings: Player mapping dictionary
    
    Returns:
        Dictionary with mapping statistics
    """
    total_mappings = 0
    confidences = []
    distances = []
    time_differences = []
    
    for frame_data in mappings.values():
        frame_mappings = frame_data.get('mappings', {})
        total_mappings += len(frame_mappings)
        
        time_differences.append(frame_data.get('time_difference', 0.0))
        
        for match_data in frame_mappings.values():
            if isinstance(match_data, dict):
                confidences.append(match_data.get('confidence', 0.0))
                distances.append(match_data.get('distance', 0.0))
    
    stats = {
        'total_frames': len(mappings),
        'total_mappings': total_mappings,
        'avg_mappings_per_frame': total_mappings / len(mappings) if mappings else 0
    }
    
    if confidences:
        stats.update({
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences),
            'high_confidence_ratio': sum(1 for c in confidences if c > 0.7) / len(confidences)
        })
    
    if distances:
        stats.update({
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances)
        })
    
    if time_differences:
        stats.update({
            'time_diff_mean': np.mean(time_differences),
            'time_diff_std': np.std(time_differences),
            'time_diff_max': np.max(time_differences)
        })
    
    return stats

def create_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for visualization.
    
    Args:
        num_colors: Number of colors to generate
    
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(num_colors):
        hue = i * 360 // num_colors
        # Convert HSV to RGB
        c = 1.0
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = 0
        
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to 0-255 range
        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
    
    return colors

def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_processing_stats(start_time: float, end_time: float, 
                             total_frames: int, processed_frames: int) -> Dict[str, Any]:
    """
    Calculate processing statistics.
    
    Args:
        start_time: Processing start time
        end_time: Processing end time
        total_frames: Total number of frames
        processed_frames: Number of processed frames
    
    Returns:
        Dictionary with processing statistics
    """
    processing_time = end_time - start_time
    fps = processed_frames / processing_time if processing_time > 0 else 0
    
    return {
        'total_processing_time': processing_time,
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'processing_fps': fps,
        'completion_rate': processed_frames / total_frames if total_frames > 0 else 0
    }

def merge_detection_windows(detections: List[Dict[str, Any]], window_size: int = 5) -> List[Dict[str, Any]]:
    """
    Merge detection results across time windows for consistency.
    
    Args:
        detections: List of detection dictionaries
        window_size: Size of the temporal window for merging
    
    Returns:
        List of merged detection dictionaries
    """
    if not detections or len(detections) < window_size:
        return detections
    
    merged_detections = []
    
    for i in range(len(detections)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(detections), i + window_size // 2 + 1)
        
        window_detections = detections[start_idx:end_idx]
        
        # Use current detection as base
        merged_detection = detections[i].copy()
        
        # Add consistency score based on neighboring frames
        if 'players' in merged_detection:
            for player_id, player_data in merged_detection['players'].items():
                consistency_count = sum(1 for det in window_detections 
                                      if 'players' in det and player_id in det['players'])
                player_data['consistency_score'] = consistency_count / len(window_detections)
        
        merged_detections.append(merged_detection)
    
    return merged_detections

def filter_low_confidence_detections(detections: Dict[str, Any], 
                                   min_confidence: float = 0.5) -> Dict[str, Any]:
    """
    Filter out low-confidence detections.
    
    Args:
        detections: Detection results dictionary
        min_confidence: Minimum confidence threshold
    
    Returns:
        Filtered detection results
    """
    filtered_detections = {}
    
    for frame_id, frame_data in detections.items():
        filtered_frame = {'players': {}}
        
        if 'players' in frame_data:
            for player_id, player_data in frame_data['players'].items():
                if player_data.get('confidence', 0) >= min_confidence:
                    filtered_frame['players'][player_id] = player_data
        
        # Copy other frame data
        for key, value in frame_data.items():
            if key != 'players':
                filtered_frame[key] = value
        
        filtered_detections[frame_id] = filtered_frame
    
    return filtered_detections

def smooth_trajectories(trajectories: Dict[str, List[Tuple[float, float]]], 
                       window_size: int = 3) -> Dict[str, List[Tuple[float, float]]]:
    """
    Smooth player trajectories using a moving average filter.
    
    Args:
        trajectories: Dictionary of player trajectories
        window_size: Size of the smoothing window
    
    Returns:
        Dictionary of smoothed trajectories
    """
    smoothed_trajectories = {}
    
    for player_id, trajectory in trajectories.items():
        if len(trajectory) <= window_size:
            smoothed_trajectories[player_id] = trajectory
            continue
        
        smoothed_trajectory = []
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            
            window_points = trajectory[start_idx:end_idx]
            
            # Calculate average position
            avg_x = sum(point[0] for point in window_points) / len(window_points)
            avg_y = sum(point[1] for point in window_points) / len(window_points)
            
            smoothed_trajectory.append((avg_x, avg_y))
        
        smoothed_trajectories[player_id] = smoothed_trajectory
    
    return smoothed_trajectories

def extract_player_crops_from_detections(video_path: str, detections: Dict[str, Any], 
                                       output_dir: str, max_crops: int = 100) -> Dict[str, List[str]]:
    """
    Extract player crops from video based on detection results for feature analysis.
    
    Args:
        video_path: Path to the video file
        detections: Detection results dictionary
        output_dir: Directory to save crops
        max_crops: Maximum number of crops to extract
    
    Returns:
        Dictionary mapping player IDs to list of crop file paths
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    player_crops = {}
    crop_count = 0
    
    for frame_idx, frame_data in detections.get('detections', {}).items():
        if crop_count >= max_crops:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        for player_id, player_data in frame_data.get('players', {}).items():
            if crop_count >= max_crops:
                break
                
            bbox = player_data.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    crop_filename = f"player_{player_id}_frame_{frame_idx}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    if player_id not in player_crops:
                        player_crops[player_id] = []
                    player_crops[player_id].append(crop_path)
                    crop_count += 1
    
    cap.release()
    return player_crops