"""
Base classes for notification system
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

from src.models.rule import Incident, Rule


@dataclass
class NotificationPayload:
    """Standard notification payload"""
    incident_id: str
    rule_id: str
    rule_description: str
    track_id: int
    
    # Timestamps
    confirmed_time: float
    
    # Evidence
    snapshot_path: Optional[str]
    video_clip_path: Optional[str]
    
    # Metadata
    avg_confidence: float
    camera_id: str = "camera_1"
    location: str = "Unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'incident_id': self.incident_id,
            'rule_id': self.rule_id,
            'rule_description': self.rule_description,
            'track_id': self.track_id,
            'confirmed_time': self.confirmed_time,
            'snapshot_path': self.snapshot_path,
            'video_clip_path': self.video_clip_path,
            'avg_confidence': self.avg_confidence,
            'camera_id': self.camera_id,
            'location': self.location
        }


class BaseNotifier(ABC):
    """Abstract base class for notifiers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit_seconds = config.get('rate_limit_seconds', 60)
        self.last_notification_time = 0
        
    @abstractmethod
    def send(self, payload: NotificationPayload) -> bool:
        """
        Send notification
        
        Args:
            payload: NotificationPayload object
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def can_send(self) -> bool:
        """Check if can send (rate limiting)"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        if current_time - self.last_notification_time < self.rate_limit_seconds:
            return False
        
        return True
    
    def mark_sent(self):
        """Mark notification as sent"""
        self.last_notification_time = time.time()
    
    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')