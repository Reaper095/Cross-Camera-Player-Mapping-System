import numpy as np
import logging
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple
import cv2

class PlayerMatcher:
    def __init__(self, similarity_threshold: float = 0.5, max_distance: float = 200.0):
        """
        Initialize player matcher with configuration
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) for valid matches
            max_distance: Maximum spatial distance (pixels) for possible matches
        """
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance  # Used in spatial similarity calculation
        self.logger = logging.getLogger(__name__)
        self.player_mappings = {}
        self.next_global_id = 1

    def match_players_temporal(self, broadcast_detections, tacticam_detections, common_frames):
        """
        Match players across cameras using temporal consistency
        
        Args:
            broadcast_detections: Dict of broadcast detections by frame
            tacticam_detections: Dict of tacticam detections by frame  
            common_frames: List of frame indices present in both videos
            
        Returns:
            Dictionary of player mappings with structure:
            {
                frame_idx: {
                    'mappings': {broadcast_id: tacticam_id},
                    'confidences': {broadcast_id: similarity_score}
                }
            }
        """
        mappings = {}  # Final result dictionary
        
        for frame_idx in common_frames:
            # Initialize frame results
            current_mappings = {
                'mappings': {},
                'confidences': {}
            }
            
            bc_frame = broadcast_detections.get(frame_idx, {'players': {}})
            tc_frame = tacticam_detections.get(frame_idx, {'players': {}})
            
            # Create similarity matrix
            bc_players = list(bc_frame['players'].items())
            tc_players = list(tc_frame['players'].items())
            similarity_matrix = np.zeros((len(bc_players), len(tc_players)))
            
            # Fill similarity matrix
            for i, (bc_id, bc_data) in enumerate(bc_players):
                for j, (tc_id, tc_data) in enumerate(tc_players):
                    similarity_matrix[i,j] = self._compute_similarity(bc_data, tc_data)
            
            # Hungarian algorithm for optimal assignment
            if similarity_matrix.size > 0:
                bc_indices, tc_indices = linear_sum_assignment(-similarity_matrix)
                
                for bc_idx, tc_idx in zip(bc_indices, tc_indices):
                    similarity = similarity_matrix[bc_idx, tc_idx]
                    if similarity > self.similarity_threshold:
                        bc_id = bc_players[bc_idx][0]
                        tc_id = tc_players[tc_idx][0]
                        current_mappings['mappings'][bc_id] = tc_id
                        current_mappings['confidences'][bc_id] = similarity
            
            mappings[frame_idx] = current_mappings
        
        return mappings

        
    def match_players_between_videos(self, broadcast_detections: pd.DataFrame,
                                   tacticam_detections: pd.DataFrame,
                                   broadcast_features: np.ndarray,
                                   tacticam_features: np.ndarray) -> Dict:
        """
        Match players between broadcast and tacticam videos
        
        Args:
            broadcast_detections: Detections from broadcast video
            tacticam_detections: Detections from tacticam video
            broadcast_features: Feature matrix for broadcast detections
            tacticam_features: Feature matrix for tacticam detections
            
        Returns:
            Dictionary with player mappings
        """
        # Find temporal overlap between videos
        broadcast_times = broadcast_detections['timestamp'].values
        tacticam_times = tacticam_detections['timestamp'].values
        
        # Find common time windows
        time_windows = self._find_temporal_windows(broadcast_times, tacticam_times)
        
        all_mappings = []
        
        for window_start, window_end in time_windows:
            # Get detections in this time window
            broadcast_window = broadcast_detections[
                (broadcast_detections['timestamp'] >= window_start) &
                (broadcast_detections['timestamp'] <= window_end)
            ]
            
            tacticam_window = tacticam_detections[
                (tacticam_detections['timestamp'] >= window_start) &
                (tacticam_detections['timestamp'] <= window_end)
            ]
            
            if len(broadcast_window) == 0 or len(tacticam_window) == 0:
                continue
            
            # Match players in this window
            window_mappings = self._match_players_in_window(
                broadcast_window, tacticam_window,
                broadcast_features, tacticam_features
            )
            
            all_mappings.extend(window_mappings)
        
        # Consolidate mappings across all windows
        final_mappings = self._consolidate_mappings(all_mappings)
        
        return final_mappings
    
    def _find_temporal_windows(self, times1: np.ndarray, times2: np.ndarray,
                             window_size: float = 3.0) -> List[Tuple[float, float]]:
        """
        Find overlapping temporal windows between two video timelines
        
        Args:
            times1: Timestamps from first video
            times2: Timestamps from second video
            window_size: Size of each window in seconds
            
        Returns:
            List of (start_time, end_time) tuples
        """
        # Find overall time range
        min_time = max(np.min(times1), np.min(times2))
        max_time = min(np.max(times1), np.max(times2))
        
        if min_time >= max_time:
            return []
        
        # Create overlapping windows
        windows = []
        current_time = min_time
        
        while current_time < max_time:
            window_end = min(current_time + window_size, max_time)
            windows.append((current_time, window_end))
            current_time += window_size / 2  # 50% overlap
        
        return windows
    

    
    def _match_players_in_window(self, broadcast_dets: pd.DataFrame,
                               tacticam_dets: pd.DataFrame,
                               broadcast_features: np.ndarray,
                               tacticam_features: np.ndarray) -> List[Dict]:
        """
        Match players within a specific time window
        
        Args:
            broadcast_dets: Broadcast detections in window
            tacticam_dets: Tacticam detections in window
            broadcast_features: Broadcast feature matrix
            tacticam_features: Tacticam feature matrix
            
        Returns:
            List of mapping dictionaries
        """
        # Group detections by approximate time
        broadcast_groups = self._group_detections_by_time(broadcast_dets, time_threshold=0.5)
        tacticam_groups = self._group_detections_by_time(tacticam_dets, time_threshold=0.5)
        
        mappings = []
        
        # Match players in each time group
        for bc_time, bc_group in broadcast_groups.items():
            # Find closest tacticam time group
            closest_tc_time = min(tacticam_groups.keys(), 
                                key=lambda x: abs(x - bc_time))
            
            if abs(closest_tc_time - bc_time) > 1.0:  # Skip if too far apart
                continue
            
            tc_group = tacticam_groups[closest_tc_time]
            
            # Compute similarity matrix
            similarity_matrix = self._compute_similarity_matrix(
                bc_group, tc_group, broadcast_features, tacticam_features
            )
            
            # Solve assignment problem
            if similarity_matrix.size > 0:
                bc_indices, tc_indices = linear_sum_assignment(-similarity_matrix)
                
                for bc_idx, tc_idx in zip(bc_indices, tc_indices):
                    similarity = similarity_matrix[bc_idx, tc_idx]
                    
                    if similarity > self.similarity_threshold:
                        mapping = {
                            'broadcast_detection': bc_group.iloc[bc_idx],
                            'tacticam_detection': tc_group.iloc[tc_idx],
                            'similarity': similarity,
                            'timestamp': bc_time
                        }
                        mappings.append(mapping)
        
        return mappings
    
    def _group_detections_by_time(self, detections: pd.DataFrame,
                                time_threshold: float = 0.5) -> Dict[float, pd.DataFrame]:
        """
        Group detections by approximate timestamp
        
        Args:
            detections: Detection DataFrame
            time_threshold: Time grouping threshold
            
        Returns:
            Dictionary mapping timestamps to grouped detections
        """
        groups = {}
        
        for timestamp in detections['timestamp'].unique():
            # Round timestamp to nearest threshold
            rounded_time = round(timestamp / time_threshold) * time_threshold
            
            if rounded_time not in groups:
                groups[rounded_time] = []
            
            groups[rounded_time].append(
                detections[detections['timestamp'] == timestamp]
            )
        
        # Concatenate groups
        final_groups = {}
        for time, group_list in groups.items():
            final_groups[time] = pd.concat(group_list, ignore_index=True)
        
        return final_groups
    
    def _compute_similarity_matrix(self, broadcast_group: pd.DataFrame,
                                 tacticam_group: pd.DataFrame,
                                 broadcast_features: np.ndarray,
                                 tacticam_features: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between two groups of detections
        
        Args:
            broadcast_group: Broadcast detection group
            tacticam_group: Tacticam detection group
            broadcast_features: Broadcast feature matrix
            tacticam_features: Tacticam feature matrix
            
        Returns:
            Similarity matrix
        """
        n_broadcast = len(broadcast_group)
        n_tacticam = len(tacticam_group)
        
        if n_broadcast == 0 or n_tacticam == 0:
            return np.array([])
        
        similarity_matrix = np.zeros((n_broadcast, n_tacticam))
        
        for i, (_, bc_det) in enumerate(broadcast_group.iterrows()):
            for j, (_, tc_det) in enumerate(tacticam_group.iterrows()):
                # Compute multi-modal similarity
                similarity = self._compute_multimodal_similarity(
                    bc_det, tc_det, broadcast_features, tacticam_features
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _compute_similarity(self, detection1: Dict, detection2: Dict) -> float:
        """
        Compute combined similarity score between two detections
        
        Args:
            detection1: First detection with features
            detection2: Second detection with features
            
        Returns:
            Combined similarity score (0-1)
        """
        try:
            # Visual similarity (cosine similarity)
            visual_sim = self._cosine_similarity(
                detection1['features']['visual'],
                detection2['features']['visual']
            )
            
            # Spatial similarity
            spatial_sim = self._compute_spatial_similarity(
                detection1['bbox'],
                detection2['bbox']
            )
            
            # Size similarity
            size_sim = self._compute_size_similarity(
                detection1['bbox'],
                detection2['bbox']
            )
            
            # Weighted combination
            weights = [0.6, 0.3, 0.1]  # visual, spatial, size
            similarities = [visual_sim, spatial_sim, size_sim]
            
            return np.average(similarities, weights=weights)
            
        except KeyError as e:
            self.logger.warning(f"Missing feature for similarity computation: {str(e)}")
            return 0.0
    
    def _compute_multimodal_similarity(self, det1: pd.Series, det2: pd.Series,
                                     features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute similarity using multiple modalities
        
        Args:
            det1: First detection
            det2: Second detection
            features1: Feature matrix for first video
            features2: Feature matrix for second video
            
        Returns:
            Combined similarity score
        """
        # Visual similarity (using feature vectors if available)
        visual_sim = 0.5  # Default value, would use actual features in practice
        
        # Spatial similarity (relative positions)
        spatial_sim = self._compute_spatial_similarity(det1, det2)
        
        # Size similarity
        size_sim = self._compute_size_similarity(det1, det2)
        
        # Motion similarity
        motion_sim = self._compute_motion_similarity(det1, det2)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # visual, spatial, size, motion
        similarities = [visual_sim, spatial_sim, size_sim, motion_sim]
        
        combined_similarity = np.average(similarities, weights=weights)
        
        return combined_similarity
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return (dot / norm) if norm > 0 else 0.0
    
    def _compute_spatial_similarity(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute spatial position similarity (0-1) using configured max_distance"""
        center1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
        center2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
        distance = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
        return max(0, 1 - distance/self.max_distance)  # Now uses instance configuration
    
    def _compute_size_similarity(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute size ratio similarity (0-1)"""
        area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        ratio = min(area1, area2) / max(area1, area2)
        return ratio if not np.isnan(ratio) else 0.0
    
    def _compute_motion_similarity(self, det1: pd.Series, det2: pd.Series) -> float:
        """
        Compute motion similarity between detections
        
        Args:
            det1: First detection
            det2: Second detection
            
        Returns:
            Motion similarity score (0-1)
        """
        # Get velocity vectors
        vel1 = np.array([det1.get('velocity_x', 0), det1.get('velocity_y', 0)])
        vel2 = np.array([det2.get('velocity_x', 0), det2.get('velocity_y', 0)])
        
        # Compute cosine similarity of velocity vectors
        if np.linalg.norm(vel1) == 0 or np.linalg.norm(vel2) == 0:
            return 0.5  # Neutral similarity for zero velocity
        
        cosine_sim = np.dot(vel1, vel2) / (np.linalg.norm(vel1) * np.linalg.norm(vel2))
        
        # Normalize to 0-1 range
        return (cosine_sim + 1) / 2
    
    def _consolidate_mappings(self, all_mappings: List[Dict]) -> Dict:
        """
        Consolidate mappings across all time windows
        
        Args:
            all_mappings: List of all mappings from different windows
            
        Returns:
            Consolidated mapping dictionary
        """
        # Group mappings by detection pairs
        mapping_groups = {}
        
        for mapping in all_mappings:
            bc_key = (mapping['broadcast_detection']['frame_idx'], 
                     tuple(mapping['broadcast_detection']['bbox']))
            tc_key = (mapping['tacticam_detection']['frame_idx'],
                     tuple(mapping['tacticam_detection']['bbox']))
            
            pair_key = (bc_key, tc_key)
            
            if pair_key not in mapping_groups:
                mapping_groups[pair_key] = []
            
            mapping_groups[pair_key].append(mapping)
        
        # Consolidate each group
        consolidated_mappings = {}
        global_id = 1
        
        for pair_key, group in mapping_groups.items():
            # Take the mapping with highest similarity
            best_mapping = max(group, key=lambda x: x['similarity'])
            
            # Assign global player ID
            consolidated_mappings[global_id] = {
                'broadcast_detection': best_mapping['broadcast_detection'],
                'tacticam_detection': best_mapping['tacticam_detection'],
                'similarity': best_mapping['similarity'],
                'confidence': len(group) / len(all_mappings)  # Confidence based on frequency
            }
            
            global_id += 1
        
        return consolidated_mappings
    
    def track_players_in_video(self, detections_df: pd.DataFrame, 
                             video_path: str) -> pd.DataFrame:
        """
        Track players within a single video to assign consistent IDs
        
        Args:
            detections_df: DataFrame with detections
            video_path: Path to video file
            
        Returns:
            DataFrame with player IDs assigned
        """
        # Sort by frame index
        df = detections_df.sort_values('frame_idx').copy()
        
        # Initialize tracking
        df['player_id'] = -1
        next_id = 1
        
        # Group by frame
        for frame_idx in df['frame_idx'].unique():
            current_frame = df[df['frame_idx'] == frame_idx]
            
            if frame_idx == df['frame_idx'].min():
                # First frame - assign new IDs
                for idx, (_, detection) in enumerate(current_frame.iterrows()):
                    df.loc[detection.name, 'player_id'] = next_id
                    next_id += 1
            else:
                # Match with previous frame
                prev_frame_idx = df[df['frame_idx'] < frame_idx]['frame_idx'].max()
                prev_frame = df[df['frame_idx'] == prev_frame_idx]
                
                # Compute similarity matrix
                similarity_matrix = self._compute_frame_similarity_matrix(
                    prev_frame, current_frame
                )
                
                # Solve assignment problem
                if similarity_matrix.size > 0:
                    prev_indices, curr_indices = linear_sum_assignment(-similarity_matrix)
                    
                    # Assign IDs based on matches
                    used_ids = set()
                    
                    for prev_idx, curr_idx in zip(prev_indices, curr_indices):
                        if similarity_matrix[prev_idx, curr_idx] > self.similarity_threshold:
                            prev_detection = prev_frame.iloc[prev_idx]
                            curr_detection = current_frame.iloc[curr_idx]
                            
                            # Assign same ID as previous detection
                            df.loc[curr_detection.name, 'player_id'] = prev_detection['player_id']
                            used_ids.add(prev_detection['player_id'])
                    
                    # Assign new IDs to unmatched detections
                    for _, detection in current_frame.iterrows():
                        if df.loc[detection.name, 'player_id'] == -1:
                            df.loc[detection.name, 'player_id'] = next_id
                            next_id += 1
        
        return df
    
    def _compute_frame_similarity_matrix(self, prev_frame: pd.DataFrame, 
                                       curr_frame: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity matrix between detections in consecutive frames
        
        Args:
            prev_frame: Detections from previous frame
            curr_frame: Detections from current frame
            
        Returns:
            Similarity matrix
        """
        n_prev = len(prev_frame)
        n_curr = len(curr_frame)
        
        if n_prev == 0 or n_curr == 0:
            return np.array([])
        
        similarity_matrix = np.zeros((n_prev, n_curr))
        
        for i, (_, prev_det) in enumerate(prev_frame.iterrows()):
            for j, (_, curr_det) in enumerate(curr_frame.iterrows()):
                # Compute similarity based on position and size
                spatial_sim = self._compute_spatial_similarity(prev_det, curr_det)
                size_sim = self._compute_size_similarity(prev_det, curr_det)
                
                # Weighted combination
                similarity = 0.7 * spatial_sim + 0.3 * size_sim
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def generate_player_mapping_report(self, mappings: Dict) -> pd.DataFrame:
        """
        Generate a report of player mappings
        
        Args:
            mappings: Dictionary with player mappings
            
        Returns:
            DataFrame with mapping report
        """
        report_data = []
        
        for global_id, mapping in mappings.items():
            report_data.append({
                'global_player_id': global_id,
                'broadcast_frame': mapping['broadcast_detection']['frame_idx'],
                'broadcast_bbox': mapping['broadcast_detection']['bbox'],
                'tacticam_frame': mapping['tacticam_detection']['frame_idx'],
                'tacticam_bbox': mapping['tacticam_detection']['bbox'],
                'similarity_score': mapping['similarity'],
                'confidence': mapping['confidence']
            })
        
        return pd.DataFrame(report_data)