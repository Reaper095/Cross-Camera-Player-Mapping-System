import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple
import pandas as pd

class PlayerDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, 
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id == 0:  # Player class
                        # Ensure crop is numpy array
                        crop = np.array(frame[
                            max(0, y1-10):min(frame.shape[0], y2+10),
                            max(0, x1-10):min(frame.shape[1], x2+10)
                        ], dtype=np.uint8)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1),
                            'crop': crop  # Ensured to be numpy array
                        })
        return detections

    def process_video(self, video_path: str, max_frames: int = None, skip_frames: int = 1) -> pd.DataFrame:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_detections = []
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and processed_frames >= max_frames):
                break
                
            if frame_idx % skip_frames == 0:
                detections = self.detect_players(frame)
                for det in detections:
                    det.update({
                        'frame_idx': frame_idx,  # Ensure consistent column name
                        'timestamp': frame_idx / fps,
                        'video_source': os.path.basename(video_path)
                    })
                    all_detections.append(det)
                processed_frames += 1
                
            frame_idx += 1
        
        cap.release()
        return pd.DataFrame(all_detections)  # Convert list of dicts to DataFrame
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], 
                           player_ids: List[int] = None) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            player_ids: Optional list of player IDs for each detection
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Color based on player ID if provided
            if player_ids and i < len(player_ids):
                player_id = player_ids[i]
                color = self._get_color_for_id(player_id)
                label = f"Player {player_id}: {conf:.2f}"
            else:
                color = (0, 255, 0)
                label = f"Player: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
    def _get_color_for_id(self, player_id: int) -> Tuple[int, int, int]:
        """
        Get consistent color for player ID
        
        Args:
            player_id: Player ID
            
        Returns:
            BGR color tuple
        """
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (255, 192, 203), (128, 128, 0), (0, 128, 128)
        ]
        return colors[player_id % len(colors)]