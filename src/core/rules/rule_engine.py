"""
Rule Engine - Evaluates rules against tracks and manages incidents
"""
from typing import List, Dict, Optional
import time
from collections import defaultdict, Counter
import uuid
import numpy as np

from src.models.detection import Track
from src.models.rule import Rule, Incident


# ──────────────────────────────────────────────────────────────
# Helper functions for helmet status inference (C-2 composition)
# ──────────────────────────────────────────────────────────────

def _get_head_region(bbox: np.ndarray, head_ratio: float = 0.40) -> np.ndarray:
    """Approximate head region: top 40% of person bbox"""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    head_height = height * head_ratio
    return np.array([x1, y1, x2, y1 + head_height])


def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou


class RuleEngine:
    """
    Evaluates monitoring rules and manages incident lifecycle.
    Supports helmetless detection via composition (person + helmet overlap).
    """
    
    def __init__(self, rules: List[Rule]):
        """
        Initialize rule engine with list of rules.
        """
        self.rules = rules
        
        # Incident management
        self.incidents: Dict[str, Incident] = {}  # incident_id -> Incident
        self.active_incidents_by_track: Dict[int, List[str]] = defaultdict(list)  # track_id -> [incident_ids]
        
        # Dwell time tracking per track per rule
        self.track_rule_timers: Dict[tuple, float] = {}  # (track_id, rule_id) -> first_timestamp
        self.track_rule_frames: Dict[tuple, List[int]] = defaultdict(list)  # (track_id, rule_id) -> [frame_ids]
        
        # Helmet status history (for temporal smoothing)
        self._helmet_history: Dict[int, List[str]] = defaultdict(list)
        
        print(f"✓ Rule Engine initialized with {len(rules)} rules")
        for rule in rules:
            print(f"  - {rule.rule_id}: {rule.description}")
    
    def infer_helmet_status(
        self,
        tracks: List[Track],
        iou_threshold: float = 0.15,
        head_ratio: float = 0.40,
        min_helmet_conf: float = 0.28,
        smooth_frames: int = 5,
        min_temporal_conf: float = 0.6
    ) -> None:
        """
        Infer helmet status for person tracks (C-2: composition method).
        Updates track.state to "helmeted", "helmetless" or "unknown".
        """
        persons = [t for t in tracks if t.class_name.lower() in {"person", "worker", "human"}]
        helmets = [
            t for t in tracks
            if any(word in t.class_name.lower() for word in 
                   {"helmet", "hard hat", "safety helmet", "hardhat", "hat", "yellow helmet"})
            and t.confidence >= min_helmet_conf
        ]

        for person in persons:
            if person.state == "lost":
                continue

            head_box = _get_head_region(person.bbox, head_ratio)
            max_iou = 0.0

            for helmet in helmets:
                iou = _compute_iou(head_box, helmet.bbox)
                if iou > max_iou:
                    max_iou = iou

            raw_status = "helmeted" if max_iou >= iou_threshold else "helmetless"

            # Temporal smoothing with majority vote
            tid = person.track_id
            self._helmet_history[tid].append(raw_status)
            if len(self._helmet_history[tid]) > smooth_frames:
                self._helmet_history[tid].pop(0)

            if self._helmet_history[tid]:
                count = Counter(self._helmet_history[tid])
                most_common_status, count_val = count.most_common(1)[0]
                temporal_conf = count_val / len(self._helmet_history[tid])
                person.state = most_common_status if temporal_conf >= min_temporal_conf else "unknown"
            else:
                person.state = "unknown"

    def evaluate(
        self,
        tracks: List[Track],
        frame_id: int,
        timestamp: float
    ) -> List[Incident]:
        """
        Evaluate all rules against current tracks and return new/updated incidents.
        """
        new_or_updated_incidents = []

        # Step 1: Infer helmet status for all person tracks (C-2)
        self.infer_helmet_status(tracks)

        # Step 2: Evaluate each rule
        for rule in self.rules:
            for track in tracks:
                if self._track_matches_rule(track, rule, frame_id, timestamp):
                    incident = self._check_dwell_time(track, rule, frame_id, timestamp)
                    if incident:
                        new_or_updated_incidents.append(incident)
                else:
                    self._reset_timer(track.track_id, rule.rule_id)

        # Step 3: Cleanup resolved incidents
        self._cleanup_resolved_incidents(timestamp)

        return new_or_updated_incidents
    
    def _track_matches_rule(
        self,
        track: Track,
        rule: Rule,
        frame_id: int,
        timestamp: float
    ) -> bool:
        """
        Check if a track matches the conditions of a rule.
        """
        # Only process tracks that are confirmed or have inferred helmet status
        if track.state not in {"confirmed", "helmeted", "helmetless", "unknown"}:
            return False
        
        if not track.detection_history:
            return False
        
        latest_detection = track.detection_history[-1]
        
        # Confidence threshold
        if latest_detection.confidence < rule.conditions.min_confidence:
            return False
        
        # Helmetless requirement (key condition for this use case)
        require_helmetless = getattr(rule.conditions, 'require_helmetless', False)
        if require_helmetless and track.state != "helmetless":
            return False
        
        # ROI check if enabled
        if rule.roi and rule.roi.enabled:
            if not rule.roi.contains_bbox(track.bbox):
                return False
        
        # Prompt match check (if applicable)
        if latest_detection.prompt_used and rule.prompt_positive not in latest_detection.prompt_used:
            return False
        
        return True
    
    def _check_dwell_time(
        self,
        track: Track,
        rule: Rule,
        frame_id: int,
        timestamp: float
    ) -> Optional[Incident]:
        """
        Check if the track has been matching the rule long enough (dwell time).
        """
        key = (track.track_id, rule.rule_id)
        
        if key not in self.track_rule_timers:
            self.track_rule_timers[key] = timestamp
            self.track_rule_frames[key] = [frame_id]
            return None
        
        self.track_rule_frames[key].append(frame_id)
        
        elapsed = timestamp - self.track_rule_timers[key]
        num_frames = len(self.track_rule_frames[key])
        
        dwell_met = elapsed >= rule.conditions.dwell_seconds
        frames_met = num_frames >= rule.conditions.min_frames
        
        if dwell_met and frames_met:
            return self._create_or_update_incident(track, rule, timestamp)
        
        return None
    
    def _create_or_update_incident(
        self,
        track: Track,
        rule: Rule,
        timestamp: float
    ) -> Incident:
        """
        Create a new incident or update an existing one for the same track-rule pair.
        """
        existing_incidents = [
            inc for inc in self.incidents.values()
            if inc.track_id == track.track_id 
            and inc.rule_id == rule.rule_id 
            and inc.state != "resolved"
        ]
        
        if existing_incidents:
            incident = existing_incidents[0]
            
            # Promote tentative → confirmed
            if incident.state == "tentative" and incident.confirmed_time is None:
                incident.confirmed_time = timestamp
                incident.state = "confirmed"
                print(f"🚨 INCIDENT CONFIRMED: {incident.incident_id} (Track {track.track_id}, Rule {rule.rule_id})")
            
            # Update metadata
            if track.detection_history:
                latest_det = track.detection_history[-1]
                incident.confidence_scores.append(latest_det.confidence)
                incident.frame_ids.append(latest_det.frame_id)
            
            return incident
        else:
            # New incident
            incident_id = f"{rule.rule_id}_{track.track_id}_{int(timestamp)}"
            
            incident = Incident(
                incident_id=incident_id,
                rule_id=rule.rule_id,
                track_id=track.track_id,
                first_detected_time=self.track_rule_timers[(track.track_id, rule.rule_id)],
                confirmed_time=None,
                resolved_time=None,
                state="tentative"
            )
            
            self.incidents[incident_id] = incident
            self.active_incidents_by_track[track.track_id].append(incident_id)
            
            print(f"⚠️ INCIDENT TENTATIVE: {incident_id} (Track {track.track_id}, Rule {rule.rule_id})")
            
            return incident
    
    def _reset_timer(self, track_id: int, rule_id: str):
        """Reset dwell timer for a track-rule combination."""
        key = (track_id, rule_id)
        self.track_rule_timers.pop(key, None)
        self.track_rule_frames.pop(key, None)
    
    def _cleanup_resolved_incidents(self, current_timestamp: float):
        """Placeholder for cleaning up old resolved incidents."""
        # Có thể thêm logic xóa incident cũ nếu cần (ví dụ > 1 giờ)
        pass
    
    def resolve_incident(self, incident_id: str, timestamp: float):
        """Manually resolve an incident."""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.state = "resolved"
            incident.resolved_time = timestamp
            print(f"✓ INCIDENT RESOLVED: {incident_id}")
    
    def should_notify(self, incident: Incident, current_timestamp: float, rule: Rule) -> bool:
        """Check if we should send notification (respect cooldown)."""
        if incident.state != "confirmed":
            return False
        
        if incident.last_notification_time is None:
            return True
        
        elapsed = current_timestamp - incident.last_notification_time
        return elapsed >= rule.actions.cooldown_seconds
    
    def mark_notified(self, incident: Incident, timestamp: float):
        """Mark that a notification was sent."""
        incident.last_notification_time = timestamp
        incident.notification_count = getattr(incident, 'notification_count', 0) + 1
    
    def get_active_incidents(self) -> List[Incident]:
        """Return all non-resolved incidents."""
        return [inc for inc in self.incidents.values() if inc.state != "resolved"]
    
    def get_confirmed_incidents(self) -> List[Incident]:
        """Return all confirmed (active) incidents."""
        return [inc for inc in self.incidents.values() if inc.state == "confirmed"]