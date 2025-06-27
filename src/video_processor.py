import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
import os
from tqdm import tqdm

class VideoProcessor:
    """
    Process videos with player tracking and generate annotated outputs
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize video processor
        
        Args:
            output_dir: Directory for output files
        """
        import logging
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_annotated_video(self, input_video_path: str, detections: pd.DataFrame, 
                            output_video_path: str = None, player_mappings: Dict = None,
                            max_frames: int = None):
        """Create annotated video with detections.
        
        Args:
            input_video_path: Path to input video file
            detections: DataFrame containing detection data with required columns
            output_video_path: Path to save annotated video
            player_mappings: Dictionary mapping player IDs between views
            max_frames: Maximum number of frames to process
        """
        # Validate detections DataFrame
        required_columns = {'frame_idx', 'bbox', 'confidence'}
        if not required_columns.issubset(detections.columns):
            missing = required_columns - set(detections.columns)
            raise ValueError(f"Detections DataFrame missing required columns: {missing}")
        
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Apply frame limit if specified
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        processed_frames = 0
        
        while processed_frames < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections for current frame
            frame_detections = detections[detections['frame_idx'] == frame_idx]
            
            if not frame_detections.empty:
                # Annotate frame directly with DataFrame
                annotated_frame = self._annotate_frame(
                    frame, 
                    frame_detections,
                    player_mappings
                )
                out.write(annotated_frame)
                processed_frames += 1
            
            frame_idx += 1
        
        cap.release()
        out.release()
        self.logger.info(f"Saved annotated video to {output_video_path}")
    
    def _annotate_frame(self, frame: np.ndarray, 
                    detections: pd.DataFrame,
                    player_mappings: Dict = None) -> np.ndarray:
        """Annotate single frame with player information from DataFrame."""
        annotated_frame = frame.copy()
        
        for _, detection in detections.iterrows():
            # Get bounding box - handle different formats
            bbox = detection['bbox']
            if isinstance(bbox, str):
                bbox = eval(bbox)  # Convert string representation to list
            elif isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get player information
            player_id = detection.get('player_id', -1)
            confidence = detection['confidence']
            
            # Draw bounding box and annotations
            color = self._get_color_for_player(player_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"Player {player_id}" if player_id != -1 else "Unknown"
            if player_mappings and player_id in player_mappings:
                label = f"P{player_mappings[player_id]}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
        
        return annotated_frame
    
    def _add_frame_info(self, frame: np.ndarray, frame_idx: int, 
                       num_detections: int) -> np.ndarray:
        """
        Add frame information overlay
        
        Args:
            frame: Input frame
            frame_idx: Current frame index
            num_detections: Number of detections in frame
            
        Returns:
            Frame with information overlay
        """
        info_text = f"Frame: {frame_idx} | Players: {num_detections}"
        
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(frame, info_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _get_color_for_player(self, player_id: int) -> Tuple[int, int, int]:
        """Get consistent color for player ID.
        
        Args:
            player_id: Should already be normalized to integer
            
        Returns:
            BGR color tuple
        """
        # Handle unknown players
        if player_id == -1:
            return (128, 128, 128)  # Gray for unknown players
        
        # Color palette
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 192, 203), # Pink
            (128, 128, 0),  # Olive
            (0, 128, 128)   # Teal
        ]
        
        return colors[player_id % len(colors)]
    
    def create_comparison_video(self, broadcast_path: str, tacticam_path: str,
                              broadcast_detections: pd.DataFrame,
                              tacticam_detections: pd.DataFrame,
                              player_mappings: Dict,
                              output_path: str,
                              max_frames: int = None) -> None:
        """
        Create side-by-side comparison video
        
        Args:
            broadcast_path: Path to broadcast video
            tacticam_path: Path to tacticam video
            broadcast_detections: Broadcast detections with IDs
            tacticam_detections: Tacticam detections with IDs
            player_mappings: Player mappings between videos
            output_path: Output video path
            max_frames: Maximum frames to process
        """
        cap1 = cv2.VideoCapture(broadcast_path)
        cap2 = cv2.VideoCapture(tacticam_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            raise ValueError("Cannot open one or both videos")
        
        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        fps = min(fps1, fps2)  # Use lower FPS
        
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Resize to same height
        target_height = min(height1, height2)
        target_width1 = int(width1 * target_height / height1)
        target_width2 = int(width2 * target_height / height2)
        
        # Setup video writer
        total_width = target_width1 + target_width2 + 10  # 10px separator
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, target_height))
        
        frame_idx = 0
        max_frames = max_frames or min(
            int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        print(f"Creating comparison video: {output_path}")
        pbar = tqdm(total=max_frames, desc="Processing comparison")
        
        while frame_idx < max_frames:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Resize frames
            frame1_resized = cv2.resize(frame1, (target_width1, target_height))
            frame2_resized = cv2.resize(frame2, (target_width2, target_height))
            
            # Get detections for current frame
            frame1_dets = broadcast_detections[
                broadcast_detections['frame_idx'] == frame_idx
            ]
            frame2_dets = tacticam_detections[
                tacticam_detections['frame_idx'] == frame_idx
            ]
            
            # Annotate frames
            frame1_annotated = self._annotate_frame(frame1_resized, frame1_dets, player_mappings)
            frame2_annotated = self._annotate_frame(frame2_resized, frame2_dets, player_mappings)
            
            # Add labels
            cv2.putText(frame1_annotated, "Broadcast", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame2_annotated, "Tacticam", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Combine frames
            separator = np.zeros((target_height, 10, 3), dtype=np.uint8)
            combined_frame = np.hstack([frame1_annotated, separator, frame2_annotated])
            
            out.write(combined_frame)
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap1.release()
        cap2.release()
        out.release()
        
        print(f"Comparison video saved: {output_path}")
    
    def export_detection_results(self, detections_df: pd.DataFrame,
                               player_mappings: Dict,
                               output_path: str) -> None:
        """
        Export detection results to CSV
        
        Args:
            detections_df: DataFrame with all detections
            player_mappings: Player mappings dictionary
            output_path: Output CSV path
        """
        # Add global player IDs if mappings available
        if player_mappings:
            detections_df['global_player_id'] = detections_df['player_id'].map(
                lambda x: player_mappings.get(x, -1)
            )
        
        # Select relevant columns
        output_columns = [
            'frame_idx', 'timestamp', 'video_source', 'player_id',
            'bbox', 'confidence', 'center_x', 'center_y', 'area',
            'velocity_x', 'velocity_y', 'speed'
        ]
        
        if 'global_player_id' in detections_df.columns:
            output_columns.append('global_player_id')
        
        # Filter existing columns
        available_columns = [col for col in output_columns if col in detections_df.columns]
        
        # Export to CSV
        detections_df[available_columns].to_csv(output_path, index=False)
        print(f"Detection results exported: {output_path}")
    
    def generate_statistics_report(self, broadcast_detections: pd.DataFrame,
                                 tacticam_detections: pd.DataFrame,
                                 player_mappings: Dict) -> Dict:
        """
        Generate statistics report
        
        Args:
            broadcast_detections: Broadcast detections
            tacticam_detections: Tacticam detections
            player_mappings: Player mappings
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'broadcast_stats': {
                'total_detections': len(broadcast_detections),
                'unique_frames': broadcast_detections['frame_idx'].nunique(),
                'unique_players': broadcast_detections['player_id'].nunique(),
                'avg_detections_per_frame': len(broadcast_detections) / broadcast_detections['frame_idx'].nunique(),
                'avg_confidence': broadcast_detections['confidence'].mean()
            },
            'tacticam_stats': {
                'total_detections': len(tacticam_detections),
                'unique_frames': tacticam_detections['frame_idx'].nunique(),
                'unique_players': tacticam_detections['player_id'].nunique(),
                'avg_detections_per_frame': len(tacticam_detections) / tacticam_detections['frame_idx'].nunique(),
                'avg_confidence': tacticam_detections['confidence'].mean()
            },
            'mapping_stats': {
                'total_mappings': len(player_mappings),
                'mapping_success_rate': len(player_mappings) / max(
                    broadcast_detections['player_id'].nunique(),
                    tacticam_detections['player_id'].nunique()
                )
            }
        }
        
        return stats