"""
rule_engine_core_v1.py
─────────────────────
Day 2 update of the rule engine core.

Changes vs original rule_engine.py:
  1. Emits structured Event objects (not just log prints) when incident confirmed.
  2. Added camera_id support throughout.
  3. Operators refactored into explicit methods:
       _op_inside_polygon()   — is track's bbox center inside ROI?
       _op_entered_polygon()  — did track just cross into ROI this frame?
       _op_dwell_time_over()  — has track been inside long enough?
       _op_near_object()      — is track within pixel-distance of another class?
       _op_attribute_missing()— does track lack a required attribute (helmet etc)?
  4. All original helmet composition logic (infer_helmet_status) preserved.
  5. Backwards-compatible: drop-in replacement for RuleEngine in run_pipeline.py.

Usage (drop-in):
    from src.core.rules.rule_engine_core_v1 import RuleEngineV1 as RuleEngine
"""

from __future__ import annotations

import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.models.detection import Track
from src.models.rule import Rule, Incident
from src.models.event import Event, EventEvidence   # ← new Day 2 model


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= x_intersect:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter + 1e-6)


def _center_distance(boxA: np.ndarray, boxB: np.ndarray) -> float:
    return float(np.linalg.norm(_bbox_center(boxA) - _bbox_center(boxB)))


def _get_head_region(bbox: np.ndarray, head_ratio: float = 0.40) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2, y1 + (y2 - y1) * head_ratio])


# ── Rule Engine ───────────────────────────────────────────────────────────────

class RuleEngineV1:
    """
    Day 2 rule engine core.
    Emits Event objects in addition to managing Incident lifecycle.
    Supports multi-rule evaluation with operator-based conditions.
    """

    # Person class aliases recognised by the engine
    PERSON_CLASSES = {"person", "worker", "human", "pedestrian"}
    # Helmet class aliases
    HELMET_CLASSES = {"helmet", "hard hat", "safety helmet", "hardhat", "hat", "yellow helmet"}

    def __init__(
        self,
        rules: List[Rule],
        camera_id: str = "cam_01",
    ):
        self.rules     = rules
        self.camera_id = camera_id

        # ── Incident state ────────────────────────────────────────────────────
        self.incidents: Dict[str, Incident] = {}
        self.active_incidents_by_track: Dict[int, List[str]] = defaultdict(list)

        # ── Dwell timers: (track_id, rule_id) → first_timestamp ──────────────
        self.track_rule_timers: Dict[Tuple, float]      = {}
        self.track_rule_frames: Dict[Tuple, List[int]]  = defaultdict(list)
        # (track_id, rule_id) -> frame_id cuối cùng còn thấy đối tượng trong vùng
        self.track_last_seen_frame: Dict[Tuple, int] = {}
        # Ngưỡng ân hạn (ví dụ: 45 frame ~ 1.5 giây nếu chạy 30fps)
        self.grace_period_frames = 60

        # ── entered_polygon: track_id → was_inside last frame ─────────────────
        # key: (track_id, rule_id) → bool
        self._was_inside: Dict[Tuple, bool] = {}

        # ── Helmet temporal smoothing ─────────────────────────────────────────
        self._helmet_history: Dict[int, List[str]] = defaultdict(list)

        # ── Event log (all events emitted this session) ───────────────────────
        self.events: List[Event] = []

        print(f"✓ RuleEngineV1 initialised  camera={camera_id}  rules={len(rules)}")
        for r in rules:
            print(f"    [{r.rule_id}] {r.description}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        tracks: List[Track],
        frame_id: int,
        timestamp: float,
    ) -> List[Incident]:
        """
        Main evaluation loop.
        Returns new/updated Incidents (same as original).
        New Events are appended to self.events.
        """
        updated_incidents: List[Incident] = []

        # Step 1 — infer helmet status (composition method, preserved from v0)
        self.infer_helmet_status(tracks)

        # Step 2 — evaluate each rule against each track
        current_active_keys = set()
        for rule in self.rules:
            for track in tracks:
                matched = self._evaluate_operators(rule, track, tracks, frame_id, timestamp)
                if matched:
                    key = (track.track_id, rule.rule_id) # Lấy key
                    current_active_keys.add(key) # GIỮ TIMER
                    incident = self._check_dwell_time(track, rule, frame_id, timestamp)
                    if incident:
                        updated_incidents.append(incident)
                        # Emit Event when newly confirmed
                        if incident.state == "confirmed":
                            event = self._emit_event(incident, track, rule, timestamp, frame_id)
                            if event:
                                self.events.append(event)
                # else:
                #     self._reset_timer(track.track_id, rule.rule_id)
        self._manage_timers(current_active_keys, frame_id)
        # Step 3 — update entered_polygon memory for next frame
        self._update_inside_memory(tracks, frame_id)

        self._cleanup_resolved_incidents(timestamp)
        return updated_incidents

    def should_notify(self, incident: Incident, current_timestamp: float, rule: Rule) -> bool:
        if incident.state != "confirmed":
            return False
        if incident.last_notification_time is None:
            return True
        return (current_timestamp - incident.last_notification_time) >= rule.actions.cooldown_seconds

    def mark_notified(self, incident: Incident, timestamp: float):
        incident.last_notification_time = timestamp
        incident.notification_count = getattr(incident, "notification_count", 0) + 1

    def get_active_incidents(self) -> List[Incident]:
        return [i for i in self.incidents.values() if i.state != "resolved"]

    def get_confirmed_incidents(self) -> List[Incident]:
        return [i for i in self.incidents.values() if i.state == "confirmed"]

    def get_recent_events(self, n: int = 20) -> List[Event]:
        """Return the N most recent events."""
        return self.events[-n:]

    def resolve_incident(self, incident_id: str, timestamp: float):
        if incident_id in self.incidents:
            inc = self.incidents[incident_id]
            inc.state = "resolved"
            inc.resolved_time = timestamp
            print(f"✓ INCIDENT RESOLVED: {incident_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Operator implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_operators(
        self,
        rule: Rule,
        track: Track,
        all_tracks: List[Track],
        frame_id: int,
        timestamp: float,
    ) -> bool:
        """
        Evaluate all applicable operators for this (rule, track) pair.
        Returns True only if ALL relevant conditions pass.
        """
        # Gate: only act on confirmed/inferred tracks with enough confidence
        if track.state not in {"confirmed", "helmeted", "helmetless", "unknown"}:
            return False
        if not track.detection_history:
            return False
        if track.detection_history[-1].confidence < rule.conditions.min_confidence:
            return False

        # ── 1. attribute_missing ──────────────────────────────────────────────
        require_helmetless = getattr(rule.conditions, "require_helmetless", False)
        if require_helmetless:
            if not self._op_attribute_missing(track, attribute="helmet"):
                return False

        # ── 2. inside_polygon / entered_polygon ───────────────────────────────
        if rule.roi and rule.roi.enabled and rule.roi.points:
            polygon = np.array(rule.roi.points)
            center  = _bbox_center(track.bbox)
            inside  = _point_in_polygon(center, polygon)

            # entered_polygon: only True on the frame the track crosses in
            key = (track.track_id, rule.rule_id)
            was_inside = self._was_inside.get(key, False)

            use_entered = getattr(rule.conditions, "use_entered_polygon", False)
            if use_entered:
                if not self._op_entered_polygon(inside, was_inside):
                    return False
            else:
                if not self._op_inside_polygon(inside):
                    return False

        # ── 3. near_object ────────────────────────────────────────────────────
        near_class    = getattr(rule.conditions, "near_class", None)
        near_px_limit = getattr(rule.conditions, "near_distance_px", None)
        if near_class and near_px_limit:
            if not self._op_near_object(track, all_tracks, near_class, near_px_limit):
                return False

        # ── 4. prompt / class name match ──────────────────────────────────────
        latest = track.detection_history[-1]
        if latest.prompt_used and rule.prompt_positive not in latest.prompt_used:
            return False

        return True

    def _op_inside_polygon(self, is_inside: bool) -> bool:
        """True when track center is currently inside the ROI polygon."""
        return is_inside

    def _op_entered_polygon(self, is_inside: bool, was_inside: bool) -> bool:
        """
        True only on the frame the track transitions outside → inside.
        Fires exactly once per entry event.
        """
        return is_inside and not was_inside

    def _op_dwell_time_over(
        self,
        track_id: int,
        rule_id: str,
        timestamp: float,
        dwell_seconds: float,
    ) -> bool:
        """True when elapsed time since first match exceeds dwell_seconds."""
        key = (track_id, rule_id)
        first_ts = self.track_rule_timers.get(key)
        if first_ts is None:
            return False
        return (timestamp - first_ts) >= dwell_seconds

    def _op_near_object(
        self,
        track: Track,
        all_tracks: List[Track],
        target_class: str,
        max_distance_px: float,
    ) -> bool:
        """
        True when track's bbox center is within max_distance_px pixels
        of any confirmed track whose class_name contains target_class.
        """
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            if target_class.lower() not in other.class_name.lower():
                continue
            if other.state not in {"confirmed", "tentative"}:
                continue
            dist = _center_distance(track.bbox, other.bbox)
            if dist <= max_distance_px:
                return True
        return False

    def _op_attribute_missing(self, track: Track, attribute: str) -> bool:
        """
        True when track is a person but lacks the given attribute.
        Currently supports attribute='helmet' using composition IoU method.
        Future: read from track.attributes dict when populated.
        """
        if attribute == "helmet":
            return track.state == "helmetless"
        # Generic fallback: check track.attributes if available
        attrs = getattr(track, "attributes", None) or {}
        return not attrs.get(attribute, True)  # missing = True triggers the rule

    # ─────────────────────────────────────────────────────────────────────────
    # Helmet inference (preserved from original rule_engine.py)
    # ─────────────────────────────────────────────────────────────────────────

    def infer_helmet_status(
        self,
        tracks: List[Track],
        iou_threshold: float = 0.15,
        head_ratio: float = 0.40,
        min_helmet_conf: float = 0.28,
        smooth_frames: int = 5,
        min_temporal_conf: float = 0.6,
    ) -> None:
        persons = [t for t in tracks if t.class_name.lower() in self.PERSON_CLASSES]
        helmets = [
            t for t in tracks
            if any(w in t.class_name.lower() for w in self.HELMET_CLASSES)
            and t.confidence >= min_helmet_conf
        ]
        for person in persons:
            if person.state == "lost":
                continue
            head_box = _get_head_region(person.bbox, head_ratio)
            max_iou  = max((_compute_iou(head_box, h.bbox) for h in helmets), default=0.0)
            raw      = "helmeted" if max_iou >= iou_threshold else "helmetless"

            tid = person.track_id
            self._helmet_history[tid].append(raw)
            if len(self._helmet_history[tid]) > smooth_frames:
                self._helmet_history[tid].pop(0)

            counts = Counter(self._helmet_history[tid])
            top, cnt = counts.most_common(1)[0]
            conf = cnt / len(self._helmet_history[tid])
            person.state = top if conf >= min_temporal_conf else "unknown"

    # ─────────────────────────────────────────────────────────────────────────
    # Dwell time + incident lifecycle (preserved + extended)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_dwell_time(
        self,
        track: Track,
        rule: Rule,
        frame_id: int,
        timestamp: float,
    ) -> Optional[Incident]:
        key = (track.track_id, rule.rule_id)
        if key not in self.track_rule_timers:
            self.track_rule_timers[key] = timestamp
            self.track_rule_frames[key] = [frame_id]
            return None

        self.track_rule_frames[key].append(frame_id)
        elapsed    = timestamp - self.track_rule_timers[key]
        num_frames = len(self.track_rule_frames[key])

        if elapsed >= rule.conditions.dwell_seconds and num_frames >= rule.conditions.min_frames:
            return self._create_or_update_incident(track, rule, timestamp)
        return None

    def _create_or_update_incident(
        self,
        track: Track,
        rule: Rule,
        timestamp: float,
    ) -> Incident:
        existing = [
            inc for inc in self.incidents.values()
            if inc.track_id == track.track_id
            and inc.rule_id == rule.rule_id
            and inc.state != "resolved"
        ]
        if existing:
            incident = existing[0]
            if incident.state == "tentative" and incident.confirmed_time is None:
                incident.confirmed_time = timestamp
                incident.state = "confirmed"
                print(f"🚨 CONFIRMED  [{incident.incident_id}]  track={track.track_id}  rule={rule.rule_id}")
            if track.detection_history:
                d = track.detection_history[-1]
                incident.confidence_scores.append(d.confidence)
                incident.frame_ids.append(d.frame_id)
            return incident

        incident_id = f"{rule.rule_id}_{track.track_id}_{int(timestamp)}"
        incident = Incident(
            incident_id=incident_id,
            rule_id=rule.rule_id,
            track_id=track.track_id,
            first_detected_time=self.track_rule_timers[(track.track_id, rule.rule_id)],
            confirmed_time=None,
            resolved_time=None,
            state="tentative",
        )
        self.incidents[incident_id] = incident
        self.active_incidents_by_track[track.track_id].append(incident_id)
        print(f"⚠️  TENTATIVE  [{incident_id}]  track={track.track_id}  rule={rule.rule_id}")
        return incident

    def _reset_timer(self, track_id: int, rule_id: str):
        key = (track_id, rule_id)
        self.track_rule_timers.pop(key, None)
        self.track_rule_frames.pop(key, None)
    def _manage_timers(self, active_track_rule_keys: set, current_frame_id: int):
        """
        Hàm mới để quét và dọn dẹp các timer đã quá hạn. (Update of func _reset_timer)
        active_track_rule_keys: Tập hợp các cặp (tid, rid) đang xuất hiện ở frame hiện tại.
        """
        # Lấy danh sách tất cả các cặp (track, rule) đang được theo dõi timer
        all_keys = list(self.track_rule_timers.keys())
        
        for key in all_keys:
            if key in active_track_rule_keys:
                # Nếu vẫn đang thấy đối tượng -> Cập nhật frame cuối cùng nhìn thấy
                self.track_last_seen_frame[key] = current_frame_id
            else:
                # Nếu KHÔNG thấy đối tượng ở frame này -> Kiểm tra xem đã mất dấu bao lâu
                last_frame = self.track_last_seen_frame.get(key, 0)
                if (current_frame_id - last_frame) > self.grace_period_frames:
                    # Đã quá thời gian ân hạn -> Reset thực sự
                    self.track_rule_timers.pop(key, None)
                    self.track_rule_frames.pop(key, None)
                    self.track_last_seen_frame.pop(key, None)
                    # (Tùy chọn) Xóa trạng thái inside để reset logic 'entered_polygon'
                    self._was_inside.pop(key, None)        

    def _update_inside_memory(self, tracks: List[Track], frame_id: int):
        """Store current inside/outside state for entered_polygon next frame."""
        for rule in self.rules:
            if not (rule.roi and rule.roi.enabled and rule.roi.points):
                continue
            polygon = np.array(rule.roi.points)
            for track in tracks:
                key    = (track.track_id, rule.rule_id)
                center = _bbox_center(track.bbox)
                self._was_inside[key] = _point_in_polygon(center, polygon)

    def _cleanup_resolved_incidents(self, current_timestamp: float):
        pass  # placeholder — can prune incidents older than N hours

    # ─────────────────────────────────────────────────────────────────────────
    # Event emission
    # ─────────────────────────────────────────────────────────────────────────

    def _emit_event(
        self,
        incident: Incident,
        track: Track,
        rule: Rule,
        timestamp: float,
        frame_id: int,
    ) -> Optional[Event]:
        """
        Create and return an Event when an incident is first confirmed.
        Avoids duplicate events for the same incident.
        """
        # Only emit once per incident confirmation
        already_emitted = any(
            e.event_id.startswith("evt_") and
            e.rule_id == rule.rule_id and
            e.track_id == track.track_id
            for e in self.events
            if abs(e.timestamp - timestamp) < 1.0
        )
        if already_emitted:
            return None

        det = track.detection_history[-1] if track.detection_history else None
        evidence = EventEvidence(
            bbox=track.bbox.tolist(),
            confidence=det.confidence if det else track.confidence,
            frame_id=frame_id,
            snapshot_path=incident.snapshots[0] if incident.snapshots else None,
            video_clip_path=incident.video_clip_path,
        )

        # Derive action string from rule's notify channels
        channels = getattr(rule.actions, "notify_channels", [])
        has_record = rule.actions.record_post_seconds > 0
        if channels and has_record:
            action = "alert+record"
        elif channels:
            action = "alert"
        else:
            action = "record"

        event = Event.create(
            rule_id=rule.rule_id,
            camera_id=self.camera_id,
            track_id=track.track_id,
            timestamp=timestamp,
            action=action,
            evidence=evidence,
            class_name=track.class_name,
            rule_description=rule.description,
        )
        print(f"📤 EVENT EMITTED  {event}")
        return event
