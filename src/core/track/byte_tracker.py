"""
ByteTrack-based multi-object tracking
"""
import numpy as np
from typing import List, Dict
import time

from src.models.detection import Detection, Track


class ByteTracker:
    """
    ByteTrack implementation for multi-object tracking
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: float = 30.0
    ):
        """
        Initialize ByteTrack tracker
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_id = 0
        
        # Try to use supervision's ByteTrack with correct API
        try:
            from supervision import ByteTrack as SVByteTrack
            
            # Check supervision version and use appropriate API
            import supervision as sv
            version = sv.__version__
            print(f"Supervision version: {version}")
            
            # New API (supervision >= 0.20.0)
            try:
                self.sv_tracker = SVByteTrack(
                    track_activation_threshold=track_thresh,
                    lost_track_buffer=track_buffer,
                    minimum_matching_threshold=match_thresh,
                    frame_rate=int(frame_rate)
                )
                print("✓ Using supervision ByteTrack (new API)")
                self.use_supervision = True
            except TypeError:
                # Old API fallback
                try:
                    self.sv_tracker = SVByteTrack(
                        track_thresh=track_thresh,
                        track_buffer=track_buffer,
                        match_thresh=match_thresh,
                        frame_rate=int(frame_rate)
                    )
                    print("✓ Using supervision ByteTrack (old API)")
                    self.use_supervision = True
                except TypeError:
                    # Very simple initialization
                    self.sv_tracker = SVByteTrack()
                    print("✓ Using supervision ByteTrack (default config)")
                    self.use_supervision = True
                    
        except ImportError:
            self.use_supervision = False
            print("⚠ supervision not found, using simple IOU tracker")
    
    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: float
    ) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            frame_id: Current frame number
            timestamp: Current timestamp
        
        Returns:
            List of active Track objects
        """
        self.frame_id = frame_id
        
        if self.use_supervision:
            return self._update_with_supervision(detections, frame_id, timestamp)
        else:
            return self._update_simple(detections, frame_id, timestamp)
    
    def _update_with_supervision(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: float
    ) -> List[Track]:
        """Update using supervision ByteTrack"""
        if not detections:
            # Mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
            return list(self.tracks.values())
        
        # Convert detections to supervision format
        det_boxes = np.array([d.bbox for d in detections])
        det_confidences = np.array([d.confidence for d in detections])
        
        # Create detections object for supervision
        from supervision import Detections as SVDetections
        sv_detections = SVDetections(
            xyxy=det_boxes,
            confidence=det_confidences
        )
        
        # Update tracker
        try:
            tracked = self.sv_tracker.update_with_detections(sv_detections)
        except Exception as e:
            print(f"Warning: Tracker update failed: {e}")
            # Fallback to simple tracking
            return self._update_simple(detections, frame_id, timestamp)
        
        # Update our Track objects
        active_track_ids = set()
        
        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for i, track_id in enumerate(tracked.tracker_id):
                active_track_ids.add(int(track_id))
                
                if track_id not in self.tracks:
                    # Create new track
                    self.tracks[track_id] = Track(
                        track_id=int(track_id),
                        bbox=tracked.xyxy[i],
                        confidence=detections[i].confidence if i < len(detections) else 0.5,
                        class_name=detections[i].class_name if i < len(detections) else "unknown",
                        first_seen_frame=frame_id,
                        last_seen_frame=frame_id,
                        first_seen_time=timestamp,
                        last_seen_time=timestamp,
                        state="tentative"
                    )
                else:
                    # Update existing track
                    det_idx = min(i, len(detections) - 1)
                    self.tracks[track_id].update(detections[det_idx], frame_id, timestamp)
        
        # Mark missed tracks
        for track_id, track in self.tracks.items():
            if track_id not in active_track_ids:
                track.mark_missed()
        
        # Remove lost tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.state != "lost"
        }
        
        return list(self.tracks.values())
    
    def _update_simple(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: float
    ) -> List[Track]:
        """Simple IOU-based tracking fallback"""
        if not detections:
            for track in self.tracks.values():
                track.mark_missed()
            return list(self.tracks.values())
        
        # Simple greedy matching based on IOU
        matched_tracks = set()
        
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._calculate_iou(detection.bbox, track.bbox)
                if iou > best_iou and iou > self.match_thresh:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].update(detection, frame_id, timestamp)
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                    state="tentative"
                )
                self.tracks[self.next_track_id] = new_track
                matched_tracks.add(self.next_track_id)
                self.next_track_id += 1
        
        # Mark unmatched tracks as missed
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.mark_missed()
        
        # Remove lost tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.state != "lost"
        }
        
        return list(self.tracks.values())
    
    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (non-lost) tracks"""
        return [track for track in self.tracks.values() if track.state != "lost"]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks"""
        return [track for track in self.tracks.values() if track.state == "confirmed"]