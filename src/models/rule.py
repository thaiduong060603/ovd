"""
Rule and Incident data models
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import json


@dataclass
class ROI:
    """Region of Interest"""
    enabled: bool
    roi_type: str  # "polygon", "rectangle", "circle"
    points: List[List[float]]  # [[x1,y1], [x2,y2], ...]
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside ROI"""
        if not self.enabled:
            return True
        
        if self.roi_type == "polygon":
            return self._point_in_polygon(point, np.array(self.points))
        elif self.roi_type == "rectangle":
            x, y = point
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            return x1 <= x <= x2 and y1 <= y <= y2
        
        return True
    
    def contains_bbox(self, bbox: np.ndarray) -> bool:
        """Check if bbox center is inside ROI"""
        # Calculate bbox center
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        return self.contains_point(center)
    
    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray casting algorithm for point in polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


@dataclass
class RuleConditions:
    """Conditions for triggering incident"""
    dwell_seconds: float
    min_confidence: float
    min_frames: int = 3
    require_helmetless: bool = False  # ← Thêm field mới, mặc định False


@dataclass
class RuleActions:
    """Actions to take when incident triggered"""
    cooldown_seconds: float
    record_pre_seconds: float
    record_post_seconds: float
    notify_channels: List[str]


@dataclass
class Rule:
    """Monitoring rule definition"""
    rule_id: str
    area_id: str
    description: str
    
    # Detection config
    prompt_positive: str
    prompt_negative: Optional[str]
    box_threshold: float
    text_threshold: float
    
    # Conditions
    conditions: RuleConditions
    
    # ROI
    roi: Optional[ROI]
    
    # Actions
    actions: RuleActions
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Rule':
        """Load rule from YAML file"""
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse ROI
        roi = None
        if 'roi' in data and data['roi'].get('enabled', False):
            roi = ROI(
                enabled=True,
                roi_type=data['roi']['type'],
                points=data['roi']['points']
            )
        
        # Parse conditions (giờ đã hỗ trợ require_helmetless)
        conditions = RuleConditions(**data['conditions'])
        
        # Parse actions
        actions = RuleActions(**data['actions'])
        
        return cls(
            rule_id=data['rule_id'],
            area_id=data['area_id'],
            description=data['description'],
            prompt_positive=data['detection']['prompt_positive'],
            prompt_negative=data['detection'].get('prompt_negative'),
            box_threshold=data['detection']['box_threshold'],
            text_threshold=data['detection']['text_threshold'],
            conditions=conditions,
            roi=roi,
            actions=actions,
            metadata=data.get('metadata', {})
        )


@dataclass
class Incident:
    """Incident record - violation of a rule"""
    incident_id: str
    rule_id: str
    track_id: int
    
    # Timestamps
    first_detected_time: float
    confirmed_time: Optional[float]
    resolved_time: Optional[float]
    
    # State machine: tentative -> confirmed -> resolved
    state: str  # "tentative", "confirmed", "resolved"
    
    # Evidence
    snapshots: List[str] = field(default_factory=list)  # Paths to snapshot images
    video_clip_path: Optional[str] = None
    
    # Metadata
    confidence_scores: List[float] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    
    # Cooldown tracking
    last_notification_time: Optional[float] = None
    notification_count: int = 0
    
    def to_json(self) -> str:
        """Export incident as JSON"""
        return json.dumps({
            'incident_id': self.incident_id,
            'rule_id': self.rule_id,
            'track_id': self.track_id,
            'state': self.state,
            'first_detected': self.first_detected_time,
            'confirmed': self.confirmed_time,
            'resolved': self.resolved_time,
            'snapshots': self.snapshots,
            'video_clip': self.video_clip_path,
            'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            'notification_count': self.notification_count
        }, indent=2)
    
    def save_metadata(self, output_path: str):
        """Save incident metadata to file"""
        with open(output_path, 'w') as f:
            f.write(self.to_json())