"""
Cross-Camera Player Mapping Project - Main Pipeline Script

This script orchestrates the complete pipeline for mapping players across
different camera views using computer vision and machine learning techniques.

Author: Anunay Minj
"""
import numpy as np
import uuid
import argparse
import pandas as pd
import time
import json
import os
import sys
from typing import Dict
from pathlib import Path
import logging

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.player_detector import PlayerDetector
from src.feature_extractor import FeatureExtractor
from src.player_matcher import PlayerMatcher
from src.video_processor import VideoProcessor
from src.utils import (
    setup_logging, create_output_directories, validate_file_paths,
    load_video_info, save_detection_results, save_mappings_to_csv,
    calculate_processing_stats, generate_timestamp
)

class PlayerMappingPipeline:
    """
    Main pipeline class for cross-camera player mapping.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.logger = setup_logging(config.get('log_level', 'INFO'))
        self.output_dirs = create_output_directories(config.get('output_dir', 'output'))
        
        # Initialize components
        self.player_detector = None
        self.feature_extractor = None
        self.player_matcher = None
        self.video_processor = None
        
        # Results storage
        self.broadcast_detections = {}
        self.tacticam_detections = {}
        self.player_mappings = {}
        self.processing_stats = {}
        
    def validate_inputs(self) -> bool:
        """
        Validate that all required input files exist.
        
        Returns:
            True if all inputs are valid, False otherwise
        """
        file_paths = {
            'Broadcast video': self.config['broadcast_path'],
            'Tacticam video': self.config['tacticam_path'],
            'YOLOv11 model': self.config['model_path']
        }
        
        return validate_file_paths(file_paths)
    
    def initialize_components(self) -> None:
        """
        Initialize all pipeline components.
        """
        self.logger.info("Initializing pipeline components...")
        
        # Initialize player detector
        self.player_detector = PlayerDetector(
            model_path=self.config['model_path'],
            confidence_threshold=self.config.get('detection_confidence', 0.3),
            iou_threshold=self.config.get('detection_iou', 0.5)
        )
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            config=self.config.get('feature_config', {})
        )
        
        # Initialize player matcher
        self.player_matcher = PlayerMatcher(
            similarity_threshold=self.config.get('matching_threshold', 0.3),
            max_distance=self.config.get('max_distance', 200)
        )
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            output_dir=self.output_dirs['visualizations']
        )
        
        self.logger.info("All components initialized successfully")
    
    def process_video_detections(self, video_path: str, video_name: str) -> dict:
        """Process detections with proper type handling and debugging"""
        self.logger.info(f"Processing detections for {video_name} video...")
        start_time = time.time()
        
        # Get video information
        video_info = load_video_info(video_path)
        self.logger.info(f"Video info: {video_info}")
        
        # Process video frames
        detections_df = self.player_detector.process_video(
            video_path=video_path,
            max_frames=self.config.get('max_frames', None),
            skip_frames=self.config.get('skip_frames', 1)
        )
        
        self.logger.info(f"Raw detections DataFrame shape: {detections_df.shape}")
        self.logger.info(f"Raw detections columns: {detections_df.columns.tolist()}")
        
        if detections_df.empty:
            self.logger.warning(f"No detections found in {video_name} video!")
            return {
                'detections': {},
                'video_info': video_info,
                'processing_stats': {'total_detections': 0, 'unique_frames': 0}
            }
        
        # Debug: Print first few detections
        self.logger.info(f"First 3 detections:\n{detections_df.head(3)}")
        
        # Convert DataFrame to proper dictionary structure
        enhanced_detections = {}
        detection_count = 0
        
        for frame_idx, frame_group in detections_df.groupby('frame_idx'):
            frame_data = {
                'frame_idx': int(frame_idx),
                'timestamp': float(frame_idx / video_info['fps']),
                'video_source': str(os.path.basename(video_path)),
                'players': {}
            }
            
            for _, detection in frame_group.iterrows():
                try:
                    # Convert Series to dictionary
                    det_dict = detection.to_dict()
                    
                    # Check if we have required fields
                    required_fields = ['bbox', 'confidence', 'class_id']
                    missing_fields = [field for field in required_fields if field not in det_dict]
                    if missing_fields:
                        self.logger.warning(f"Missing required fields: {missing_fields}")
                        continue
                    
                    # Extract features if crop is available
                    features = {}
                    if 'crop' in det_dict and det_dict['crop'] is not None:
                        try:
                            features = self.feature_extractor.extract_features(
                                det_dict['crop'],
                                det_dict['bbox'],
                                video_info
                            )
                        except Exception as e:
                            self.logger.warning(f"Feature extraction failed: {str(e)}")
                            features = {}
                    
                    # Calculate center and area from bbox
                    bbox = det_dict['bbox']
                    if isinstance(bbox, str):
                        bbox = eval(bbox)
                    elif isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    
                    x1, y1, x2, y2 = bbox
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Create player data
                    player_data = {
                        'bbox': bbox,
                        'confidence': float(det_dict['confidence']),
                        'class_id': int(det_dict['class_id']),
                        'center': center,
                        'area': float(area),
                        'features': features
                    }
                    
                    # Generate a simple player ID based on detection order
                    player_id = f"player_{detection_count}"
                    frame_data['players'][player_id] = player_data
                    detection_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing detection in frame {frame_idx}: {str(e)}")
                    continue
                    
            if frame_data['players']:  # Only add frames with valid detections
                enhanced_detections[frame_idx] = frame_data
        
        # Log detection statistics
        total_detections = sum(len(frame_data['players']) for frame_data in enhanced_detections.values())
        self.logger.info(f"Processed {len(enhanced_detections)} frames with {total_detections} total detections")
        
        # Calculate and return results
        stats = calculate_processing_stats(
            start_time, time.time(), 
            video_info['frame_count'], 
            len(enhanced_detections)
        )
        stats['total_detections'] = total_detections
        
        # Save results
        detection_file = os.path.join(
            self.output_dirs['detections'], 
            f"{video_name}_detections_{generate_timestamp()}.json"
        )
        save_detection_results(enhanced_detections, detection_file)
        
        return {
            'detections': enhanced_detections,
            'video_info': video_info,
            'processing_stats': stats
        }

    
    def match_players_across_cameras(self) -> dict:
        """
        Match players between broadcast and tacticam videos with proper confidence calculation.
        """
        self.logger.info("Matching players across cameras...")

        # Check detections
        if not self.broadcast_detections.get('detections'):
            self.logger.error("No broadcast detections available for matching!")
            return {}

        if not self.tacticam_detections.get('detections'):
            self.logger.error("No tacticam detections available for matching!")
            return {}

        start_time = time.time()
        time_tolerance = self.config.get("time_tolerance", 1.0)
        max_distance = self.config.get('max_distance', 1200)

        # Convert to lists for timestamp-based alignment
        broadcast_frames = list(self.broadcast_detections['detections'].values())
        tacticam_frames = list(self.tacticam_detections['detections'].values())

        # DEBUG: Print timestamp ranges
        b_timestamps = [f['timestamp'] for f in broadcast_frames if f['players']]
        t_timestamps = [f['timestamp'] for f in tacticam_frames if f['players']]
        
        if b_timestamps and t_timestamps:
            self.logger.info(f"Broadcast timestamps: {min(b_timestamps):.3f} to {max(b_timestamps):.3f}")
            self.logger.info(f"Tacticam timestamps: {min(t_timestamps):.3f} to {max(t_timestamps):.3f}")
            self.logger.info(f"Using time tolerance: {time_tolerance}s")

        mappings = {}
        total_matches = 0

        # Match each broadcast frame with closest timestamp in tacticam
        for b_frame in broadcast_frames:
            b_ts = b_frame['timestamp']
            b_players = b_frame['players']
            
            if not b_players:  # Skip frames without players
                continue
                
            matched_t_frame = None
            best_time_diff = float('inf')

            # Find matching tacticam frame based on timestamp
            for t_frame in tacticam_frames:
                if not t_frame['players']:  # Skip frames without players
                    continue
                    
                t_ts = t_frame['timestamp']
                time_diff = abs(t_frame['timestamp'] - b_ts)
                
                if time_diff <= time_tolerance and time_diff < best_time_diff:
                    matched_t_frame = t_frame
                    best_time_diff = time_diff

            if not matched_t_frame:
                self.logger.debug(f"No tacticam frame found for broadcast ts={b_ts:.3f}")
                continue

            t_players = matched_t_frame['players']
            self.logger.debug(f"Matching b_ts={b_ts:.3f} with t_ts={matched_t_frame['timestamp']:.3f} "
                            f"(diff={best_time_diff:.3f}s)")
            self.logger.debug(f"Players: broadcast={len(b_players)}, tacticam={len(t_players)}")
            
            frame_mappings = {}

            # Match players within this frame pair with confidence calculation
            for b_id, b_data in b_players.items():
                b_center = b_data['center']
                b_confidence = b_data.get('confidence', 0.0)
                b_features = b_data.get('features', {})
                
                best_match = None
                best_distance = float('inf')
                best_match_confidence = 0.0

                for t_id, t_data in t_players.items():
                    t_center = t_data['center']
                    t_confidence = t_data.get('confidence', 0.0)
                    t_features = t_data.get('features', {})
                    
                    # Calculate spatial distance
                    distance = np.sqrt((b_center[0] - t_center[0]) ** 2 + (b_center[1] - t_center[1]) ** 2)
                    
                    if distance < max_distance:
                        # Calculate comprehensive confidence score
                        match_confidence = self._calculate_match_confidence(
                            b_data, t_data, distance, max_distance, best_time_diff, time_tolerance
                        )
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = t_id
                            best_match_confidence = match_confidence

                if best_match and best_match_confidence > 0.1:  # Minimum confidence threshold
                    frame_mappings[b_id] = {
                        'tacticam_id': best_match,
                        'confidence': best_match_confidence,
                        'distance': best_distance,
                        'spatial_confidence': max(0, 1 - (best_distance / max_distance))
                    }
                    total_matches += 1
                    self.logger.debug(f"Matched {b_id} -> {best_match} "
                                    f"(distance: {best_distance:.1f}px, confidence: {best_match_confidence:.3f})")

            if frame_mappings:
                mappings[b_frame['frame_idx']] = {
                    'mappings': frame_mappings,
                    'timestamp': b_ts,
                    'tacticam_timestamp': matched_t_frame['timestamp'],
                    'time_difference': best_time_diff
                }

        end_time = time.time()
        self.logger.info(f"Player matching completed - Generated {total_matches} matches "
                        f"across {len(mappings)} frames in {end_time - start_time:.2f}s")

        # Save to CSV with proper confidence values
        if mappings:
            mappings_file = os.path.join(
                self.output_dirs['results'],
                f"player_mappings_{generate_timestamp()}.csv"
            )
            self._save_enhanced_mappings_to_csv(mappings, mappings_file)

        return mappings

    def _calculate_match_confidence(self, b_data, t_data, distance, max_distance, time_diff, time_tolerance):
        """
        Calculate comprehensive confidence score for player matching.
        
        Args:
            b_data: Broadcast player data
            t_data: Tacticam player data  
            distance: Spatial distance between players
            max_distance: Maximum allowed distance
            time_diff: Time difference between frames
            time_tolerance: Maximum allowed time difference
            
        Returns:
            Confidence score between 0 and 1
        """
        # Spatial confidence (closer = higher confidence)
        spatial_conf = max(0, 1 - (distance / max_distance))
        
        # Temporal confidence (closer in time = higher confidence)
        temporal_conf = max(0, 1 - (time_diff / time_tolerance))
        
        # Detection confidence (average of both detections)
        detection_conf = (b_data.get('confidence', 0.0) + t_data.get('confidence', 0.0)) / 2
        
        # Feature similarity confidence (if features available)
        feature_conf = 0.5  # Default neutral value
        b_features = b_data.get('features', {})
        t_features = t_data.get('features', {})
        
        if b_features and t_features:
            feature_conf = self._calculate_feature_similarity(b_features, t_features)
        
        # Size similarity confidence
        b_area = b_data.get('area', 0)
        t_area = t_data.get('area', 0)
        if b_area > 0 and t_area > 0:
            size_ratio = min(b_area, t_area) / max(b_area, t_area)
            size_conf = size_ratio  # Similar sizes get higher confidence
        else:
            size_conf = 0.5  # Default neutral value
        
        # Weighted combination of all confidence factors
        weights = {
            'spatial': 0.4,      # Most important: spatial proximity
            'detection': 0.25,   # Detection quality
            'temporal': 0.15,    # Time synchronization
            'feature': 0.1,      # Feature similarity
            'size': 0.1         # Size similarity
        }
        
        total_confidence = (
            weights['spatial'] * spatial_conf +
            weights['detection'] * detection_conf +
            weights['temporal'] * temporal_conf +
            weights['feature'] * feature_conf +
            weights['size'] * size_conf
        )
        
        return min(1.0, max(0.0, total_confidence))  # Clamp between 0 and 1

    def _calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature dictionaries."""
        try:
            # Color histogram similarity
            color_sim = 0.0
            if 'color_hist' in features1 and 'color_hist' in features2:
                hist1 = np.array(features1['color_hist'])
                hist2 = np.array(features2['color_hist'])
                if len(hist1) == len(hist2):
                    # Use correlation coefficient for histogram comparison
                    correlation = np.corrcoef(hist1, hist2)[0, 1]
                    color_sim = max(0, correlation) if not np.isnan(correlation) else 0.0
            
            # HOG feature similarity (if available)
            hog_sim = 0.0
            if 'hog' in features1 and 'hog' in features2:
                hog1 = np.array(features1['hog']).flatten()
                hog2 = np.array(features2['hog']).flatten()
                if len(hog1) == len(hog2) and len(hog1) > 0:
                    # Cosine similarity for HOG features
                    dot_product = np.dot(hog1, hog2)
                    norms = np.linalg.norm(hog1) * np.linalg.norm(hog2)
                    if norms > 0:
                        hog_sim = max(0, dot_product / norms)
            
            # Average the similarities
            similarities = [s for s in [color_sim, hog_sim] if s > 0]
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            self.logger.warning(f"Feature similarity calculation failed: {e}")
            return 0.5  # Return neutral value on error

    def _save_enhanced_mappings_to_csv(self, mappings, filepath):
        """Save mappings to CSV with proper confidence values."""
        rows = []
        
        for frame_idx, frame_data in mappings.items():
            for b_id, match_data in frame_data['mappings'].items():
                if isinstance(match_data, dict):
                    # New format with confidence
                    t_id = match_data['tacticam_id']
                    confidence = match_data['confidence']
                else:
                    # Old format (just the tacticam ID)
                    t_id = match_data
                    confidence = 0.0
                    
                rows.append({
                    'frame_id': frame_idx,
                    'broadcast_player_id': b_id,
                    'tacticam_player_id': t_id,
                    'confidence': confidence,
                    'timestamp': frame_data.get('timestamp', 0.0),
                    'time_difference': frame_data.get('time_difference', 0.0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Enhanced mappings saved to: {filepath}")
        
        # Log confidence statistics
        if not df.empty:
            self.logger.info(f"Confidence statistics - Mean: {df['confidence'].mean():.3f}, "
                            f"Min: {df['confidence'].min():.3f}, Max: {df['confidence'].max():.3f}")

    
    def _player_id_to_int(self, player_id) -> int:
        """Convert various player ID formats to consistent integers.
        
        Args:
            player_id: Could be int, str (hex/uuid), or other format
            
        Returns:
            Consistent integer representation
        """
        if isinstance(player_id, int):
            return player_id
        try:
            # Try direct conversion for numeric strings
            return int(player_id)
        except ValueError:
            try:
                # Try hexadecimal conversion
                return int(player_id, 16)
            except ValueError:
                # Fallback to hash for other string formats
                return abs(hash(player_id)) % (10**8)  # Limit to 8-digit number
            
    def _normalize_player_id(self, player_id) -> int:
        """Convert any player ID format to consistent integer.
        
        Handles:
        - Integers
        - Hexadecimal strings (like 'd567715b')
        - UUID strings
        - Other string formats
        
        Returns:
            Consistent integer representation
        """
        if isinstance(player_id, int):
            return player_id
        
        if isinstance(player_id, str):
            # Try hexadecimal conversion first
            try:
                return int(player_id, 16)
            except ValueError:
                # If not hex, try regular integer
                try:
                    return int(player_id)
                except ValueError:
                    # Fallback to hash for other strings
                    return abs(hash(player_id)) % (10**8)  # 8-digit limit
        
        # For any other type (float, etc.)
        try:
            return int(player_id)
        except (ValueError, TypeError):
            return abs(hash(str(player_id))) % (10**8)
    
    def generate_visualizations(self) -> None:
        """Generate annotated videos and visualizations."""
        self.logger.info("Generating visualizations...")
        start_time = time.time()
        
        def create_detection_df(detections_dict):
            """Helper function to create DataFrame from detections dictionary"""
            records = []
            for frame_idx, frame_data in detections_dict.items():
                for player_id, player_data in frame_data['players'].items():
                    # Use the normalize method to handle different player ID formats
                    numeric_id = self._normalize_player_id(player_id)
                    
                    record = {
                        'frame_idx': frame_idx,
                        'player_id': numeric_id,
                        'bbox': player_data['bbox'],
                        'confidence': player_data['confidence'],
                        'center': player_data.get('center', [0, 0]),
                        'area': player_data.get('area', 0)
                    }
                    records.append(record)
            return pd.DataFrame(records)

        # Create DataFrames using the helper function
        broadcast_df = create_detection_df(self.broadcast_detections['detections'])
        tacticam_df = create_detection_df(self.tacticam_detections['detections'])

        # Generate individual annotated videos
        self.logger.info("Creating individual annotated videos...")
        
        broadcast_output = os.path.join(
            self.output_dirs['visualizations'], 
            f"broadcast_annotated_{generate_timestamp()}.mp4"
        )
        
        self.video_processor.create_annotated_video(
            input_video_path=self.config['broadcast_path'],
            detections=broadcast_df,
            output_video_path=broadcast_output,
            player_mappings=self._create_simple_mapping_dict(),
            max_frames=self.config.get('max_frames')
        )
        
        tacticam_output = os.path.join(
            self.output_dirs['visualizations'], 
            f"tacticam_annotated_{generate_timestamp()}.mp4"
        )
        
        self.video_processor.create_annotated_video(
            input_video_path=self.config['tacticam_path'],
            detections=tacticam_df,
            output_video_path=tacticam_output,
            player_mappings=self._create_simple_mapping_dict(),
            max_frames=self.config.get('max_frames')
        )
        
        # Generate side-by-side comparison video
        self.logger.info("Creating side-by-side comparison video...")
        
        comparison_output = os.path.join(
            self.output_dirs['visualizations'], 
            f"comparison_video_{generate_timestamp()}.mp4"
        )
        
        try:
            self.video_processor.create_comparison_video(
                broadcast_path=self.config['broadcast_path'],
                tacticam_path=self.config['tacticam_path'],
                broadcast_detections=broadcast_df,
                tacticam_detections=tacticam_df,
                player_mappings=self._create_simple_mapping_dict(),
                output_path=comparison_output,
                max_frames=self.config.get('max_frames')
            )
            self.logger.info(f"Comparison video saved: {comparison_output}")
        except Exception as e:
            self.logger.warning(f"Failed to create comparison video: {str(e)}")
        
        # Generate summary visualization with statistics
        self._create_detection_summary_video(broadcast_df, tacticam_df)
        
        end_time = time.time()
        self.logger.info(f"Visualization generation completed in {end_time - start_time:.2f}s")


    def _create_simple_mapping_dict(self) -> Dict:
        """Create a simplified mapping dictionary for visualization."""
        simple_mappings = {}
        
        for frame_idx, frame_mappings in self.player_mappings.items():
            if 'mappings' in frame_mappings:
                for broadcast_id, tacticam_id in frame_mappings['mappings'].items():
                    # Normalize both IDs
                    broadcast_norm = self._normalize_player_id(broadcast_id)
                    tacticam_norm = self._normalize_player_id(tacticam_id)
                    simple_mappings[broadcast_norm] = tacticam_norm
                    simple_mappings[tacticam_norm] = broadcast_norm
        
        return simple_mappings

    def _create_detection_summary_video(self, broadcast_df: pd.DataFrame, tacticam_df: pd.DataFrame) -> None:
        """Create a summary video with detection statistics overlay."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.patches import Rectangle
            
            self.logger.info("Creating detection summary visualization...")
            
            # Get frame statistics
            broadcast_stats = broadcast_df.groupby('frame_idx').agg({
                'player_id': 'count',
                'confidence': 'mean'
            }).rename(columns={'player_id': 'detections', 'confidence': 'avg_confidence'})
            
            tacticam_stats = tacticam_df.groupby('frame_idx').agg({
                'player_id': 'count', 
                'confidence': 'mean'
            }).rename(columns={'player_id': 'detections', 'confidence': 'avg_confidence'})
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot detection counts over time
            ax1.plot(broadcast_stats.index, broadcast_stats['detections'], 
                    label='Broadcast View', color='blue', linewidth=2)
            ax1.plot(tacticam_stats.index, tacticam_stats['detections'], 
                    label='Tactical View', color='red', linewidth=2)
            ax1.set_ylabel('Number of Detections')
            ax1.set_title('Player Detections Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot confidence scores over time
            ax2.plot(broadcast_stats.index, broadcast_stats['avg_confidence'], 
                    label='Broadcast View', color='blue', linewidth=2)
            ax2.plot(tacticam_stats.index, tacticam_stats['avg_confidence'], 
                    label='Tactical View', color='red', linewidth=2)
            ax2.set_ylabel('Average Confidence')
            ax2.set_xlabel('Frame Number')
            ax2.set_title('Average Detection Confidence Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save summary plot
            summary_plot_path = os.path.join(
                self.output_dirs['visualizations'], 
                f"detection_summary_{generate_timestamp()}.png"
            )
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Detection summary plot saved: {summary_plot_path}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping summary visualization")
        except Exception as e:
            self.logger.warning(f"Failed to create summary visualization: {str(e)}")

        
    
    
    def generate_final_report(self) -> None:
        """
        Generate final processing report and statistics.
        """
        self.logger.info("Generating final report...")
        
        # Compile comprehensive statistics
        stats = {
            'processing_timestamp': generate_timestamp(),
            'input_files': {
                'broadcast_video': self.config['broadcast_path'],
                'tacticam_video': self.config['tacticam_path'],
                'model_file': self.config['model_path']
            },
            'video_info': {
                'broadcast': self.broadcast_detections.get('video_info', {}),
                'tacticam': self.tacticam_detections.get('video_info', {})
            },
            'processing_stats': {
                'broadcast': self.broadcast_detections.get('processing_stats', {}),
                'tacticam': self.tacticam_detections.get('processing_stats', {})
            },
            'matching_stats': {
                'total_frames_matched': len(self.player_mappings),
                'total_player_matches': sum(len(frame_mappings.get('mappings', {})) 
                                          for frame_mappings in self.player_mappings.values())
            },
            'output_files': {
                'results_directory': self.output_dirs['results'],
                'visualizations_directory': self.output_dirs['visualizations'],
                'detections_directory': self.output_dirs['detections']
            }
        }
        
        # Save comprehensive report
        report_file = os.path.join(
            self.output_dirs['results'], 
            f"detection_stats_{generate_timestamp()}.json"
        )
        
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*60)
        print("CROSS-CAMERA PLAYER MAPPING - PROCESSING COMPLETE")
        print("="*60)
        print(f"Broadcast video: {stats['input_files']['broadcast_video']}")
        print(f"Tacticam video: {stats['input_files']['tacticam_video']}")
        print(f"Model used: {stats['input_files']['model_file']}")
        print(f"Total frames matched: {stats['matching_stats']['total_frames_matched']}")
        print(f"Total player matches: {stats['matching_stats']['total_player_matches']}")
        print(f"Output directory: {self.output_dirs['base']}")
        print(f"Report saved: {report_file}")
        print("="*60)
    
    def run(self) -> None:
        """
        Execute the complete pipeline.
        """
        try:
            self.logger.info("Starting Cross-Camera Player Mapping Pipeline...")
            
            # Validate inputs
            if not self.validate_inputs():
                self.logger.error("Input validation failed. Exiting.")
                return
            
            # Initialize components
            self.initialize_components()
            
            # Process both videos
            self.broadcast_detections = self.process_video_detections(
                self.config['broadcast_path'], 'broadcast'
            )
            
            self.tacticam_detections = self.process_video_detections(
                self.config['tacticam_path'], 'tacticam'
            )
            
            # Match players across cameras
            self.player_mappings = self.match_players_across_cameras()
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Generate final report
            self.generate_final_report()
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise

        finally:
            # Clean up resources if needed
            self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Clean up any resources used by the pipeline."""
        self.logger.info("Cleaning up pipeline resources...")

def main():
    """Main function to run the pipeline from command line."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Cross-Camera Player Mapping Pipeline'
    )
    
    # Required arguments
    parser.add_argument('--broadcast_path', required=True,
                       help='Path to broadcast video file')
    parser.add_argument('--tacticam_path', required=True,
                       help='Path to tacticam video file')
    parser.add_argument('--model_path', required=True,
                       help='Path to YOLOv11 model file')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='output',
                       help='Output directory path')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--max_frames', type=int,
                       help='Maximum number of frames to process')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Number of frames to skip between processing')
    parser.add_argument('--detection_confidence', type=float, default=0.5,
                       help='Confidence threshold for player detection')
    parser.add_argument('--detection_iou', type=float, default=0.5,
                       help='IOU threshold for NMS in detection')
    parser.add_argument('--matching_threshold', type=float, default=0.3,
                       help='Similarity threshold for player matching')
    parser.add_argument('--max_distance', type=float, default=200.0,
                       help='Maximum distance for spatial matching')
    
    args = parser.parse_args()

    # Create configuration dictionary
    config = {
        'broadcast_path': args.broadcast_path,
        'tacticam_path': args.tacticam_path,
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'log_level': args.log_level,
        'max_frames': args.max_frames,
        'skip_frames': args.skip_frames,
        'detection_confidence': args.detection_confidence,
        'detection_iou': args.detection_iou,
        'matching_threshold': args.matching_threshold,
        #'max_distance': args.max_distance,
        'time_tolerance': 1.0,  # Increase from 0.08 to 1.0 seconds
        'max_distance': 1200,   # Increase from 200 to 1200 pixels
        'feature_config': {
            'color_hist_bins': 32,
            'hog_orientations': 9,
            'hog_pixels_per_cell': (8, 8),
            'hog_cells_per_block': (2, 2),
            #'motion_history_length': 5
        }
    }

    # Initialize and run the pipeline
    pipeline = PlayerMappingPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()