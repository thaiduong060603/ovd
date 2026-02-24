"""
Data models for detection and tracking
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from datetime import datetime


@dataclass
class Detection:
    """Single object detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2] format
    confidence: float
    class_name: str
    prompt_used: str
    frame_id: int = 0
    
    def __post_init__(self):
        """Validate bbox format"""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
    
    @property
    def center(self) -> np.ndarray:
        """Get center point of bbox"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    @property
    def area(self) -> float:
        """Get bbox area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class Track:
    """Tracked object across frames"""
    track_id: int
    bbox: np.ndarray  # Current bbox
    confidence: float
    class_name: str
    
    # Tracking metadata
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float  # timestamp
    last_seen_time: float
    
    # Track state
    state: str = "tentative"  # tentative, confirmed, lost
    hit_streak: int = 0  # Consecutive frames detected
    time_since_update: int = 0  # Frames since last detection
    
    # History
    detection_history: List[Detection] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate state"""
        valid_states = ["tentative", "confirmed", "lost"]
        if self.state not in valid_states:
            raise ValueError(f"state must be one of {valid_states}")
    
    @property
    def age(self) -> int:
        """Number of frames this track has existed"""
        return self.last_seen_frame - self.first_seen_frame + 1
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return self.last_seen_time - self.first_seen_time
    
    def update(self, detection: Detection, frame_id: int, timestamp: float):
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_seen_frame = frame_id
        self.last_seen_time = timestamp
        self.hit_streak += 1
        self.time_since_update = 0
        self.detection_history.append(detection)
        
        # Promote to confirmed after 3 consecutive hits
        if self.hit_streak >= 3 and self.state == "tentative":
            self.state = "confirmed"
    
    def mark_missed(self):
        """Mark that this track was not detected in current frame"""
        self.time_since_update += 1
        self.hit_streak = 0
        
        # Mark as lost after 30 frames without detection
        if self.time_since_update > 30:
            self.state = "lost"