"""
Visualization utilities for detections and tracks
"""
import cv2
import numpy as np
from typing import List
import colorsys

from src.models.detection import Detection, Track


class Visualizer:
    """Visualize detections and tracks on frames"""
    
    def __init__(self):
        # Generate distinct colors for tracks
        self.track_colors = {}
        
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: tuple = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of Detection objects
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            
            # Draw bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(
                frame_copy,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame_copy
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        show_id: bool = True,
        show_state: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw tracked objects on frame
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            show_id: Show track ID
            show_state: Show track state
            thickness: Line thickness
        
        Returns:
            Frame with drawn tracks
        """
        frame_copy = frame.copy()
        
        for track in tracks:
            # Get consistent color for this track
            color = self._get_track_color(track.track_id)
            
            x1, y1, x2, y2 = track.bbox.astype(int)
            
            # Different style for different states
            if track.state == "tentative":
                line_type = cv2.LINE_4  # Dotted effect
                thickness_adj = thickness - 1
            elif track.state == "confirmed":
                line_type = cv2.LINE_AA
                thickness_adj = thickness
            else:  # lost
                line_type = cv2.LINE_4
                thickness_adj = 1
                color = (128, 128, 128)  # Gray
            
            # Draw bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness_adj, line_type)
            
            # Build label
            label_parts = []
            if show_id:
                label_parts.append(f"ID:{track.track_id}")
            label_parts.append(f"{track.class_name}")
            if show_state:
                label_parts.append(f"[{track.state}]")
            label_parts.append(f"{track.confidence:.2f}")
            
            label = " ".join(label_parts)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(
                frame_copy,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame_copy,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Draw trajectory (last 30 positions)
            if len(track.detection_history) > 1:
                points = [det.center.astype(int) for det in track.detection_history[-30:]]
                for i in range(len(points) - 1):
                    cv2.line(frame_copy, tuple(points[i]), tuple(points[i + 1]), color, 2)
        
        return frame_copy
    
    def _get_track_color(self, track_id: int) -> tuple:
        """Get consistent color for track ID"""
        if track_id not in self.track_colors:
            # Generate color using HSV for better distribution
            hue = (track_id * 37) % 360  # Prime number for good distribution
            rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.95)
            bgr = tuple(int(c * 255) for c in reversed(rgb))
            self.track_colors[track_id] = bgr
        
        return self.track_colors[track_id]
    
    def add_info_panel(
        self,
        frame: np.ndarray,
        frame_id: int,
        fps: float,
        num_detections: int,
        num_tracks: int
    ) -> np.ndarray:
        """Add information panel to frame"""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
        
        # Info text
        info = [
            f"Frame: {frame_id}",
            f"FPS: {fps:.1f}",
            f"Detections: {num_detections}",
            f"Tracks: {num_tracks}"
        ]
        
        y = 35
        for line in info:
            cv2.putText(
                frame_copy,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y += 25
        
        return frame_copy
    def draw_roi(self, frame: np.ndarray, roi: 'ROI') -> np.ndarray:
        """Draw Region of Interest on frame"""
        if not roi.enabled:
            return frame
        
        overlay = frame.copy()
        
        if roi.roi_type == "polygon":
            points = np.array(roi.points, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            cv2.polylines(frame, [points], True, (0, 255, 0), 3)
        elif roi.roi_type == "rectangle":
            x1, y1 = map(int, roi.points[0])
            x2, y2 = map(int, roi.points[1])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Blend with transparency
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # ROI label
        cv2.putText(
            frame,
            "ROI",
            (int(roi.points[0][0]) + 10, int(roi.points[0][1]) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def draw_incidents(
        self,
        frame: np.ndarray,
        incidents: List['Incident'],
        tracks: List[Track]
    ) -> np.ndarray:
        """Draw incident markers on tracks"""
        # Create track_id to incident map
        track_incidents = {inc.track_id: inc for inc in incidents}
        
        for track in tracks:
            if track.track_id in track_incidents:
                incident = track_incidents[track.track_id]
                x1, y1, x2, y2 = track.bbox.astype(int)
                
                # Draw incident indicator
                if incident.state == "confirmed":
                    color = (0, 0, 255)  # Red
                    label = "⚠️ INCIDENT"
                else:
                    color = (0, 165, 255)  # Orange
                    label = "⚠️ TENTATIVE"
                
                # Draw warning box
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), color, 3)
                
                # Draw warning label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(
                    frame,
                    (x1, y2 + 5),
                    (x1 + label_size[0] + 10, y2 + label_size[1] + 15),
                    color,
                    -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y2 + label_size[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        return frame
    
    def add_info_panel_with_incidents(
        self,
        frame: np.ndarray,
        frame_id: int,
        fps: float,
        num_detections: int,
        num_tracks: int,
        num_incidents: int,
        num_confirmed: int
    ) -> np.ndarray:
        """Enhanced info panel with incident counts"""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
        
        # Info text
        info = [
            f"Frame: {frame_id}",
            f"FPS: {fps:.1f}",
            f"Detections: {num_detections}",
            f"Tracks: {num_tracks}",
            f"Incidents: {num_incidents}",
            f"Confirmed: {num_confirmed}"
        ]
        
        y = 35
        for i, line in enumerate(info):
            # Highlight incidents in red
            color = (0, 0, 255) if i >= 4 and num_confirmed > 0 else (255, 255, 255)
            cv2.putText(
                frame_copy,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            y += 27
        
        return frame_copy