"""
Event output model — emitted by RuleEngine when a rule condition is confirmed.

This is the structured output contract between the rule engine and downstream
consumers (notification manager, incident recorder, API, storage, GUI).

Defined for Day 2 deliverable: event_output_schema_v1
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
import json
import numpy as np
from datetime import datetime, timezone


# ── Evidence ──────────────────────────────────────────────────────────────────

@dataclass
class EventEvidence:
    """
    Visual/spatial evidence attached to an event.
    Carries enough data for an operator to review what triggered the rule.
    """
    bbox: List[float]                       # [x1, y1, x2, y2] at trigger frame
    confidence: float                       # detection confidence at trigger
    frame_id: int                           # frame index when event was confirmed
    snapshot_path: Optional[str] = None    # path to saved snapshot image
    video_clip_path: Optional[str] = None  # path to saved video clip
    extra: Dict[str, Any] = field(default_factory=dict)  # future: pose, attributes...

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "frame_id": self.frame_id,
            "snapshot_path": self.snapshot_path,
            "video_clip_path": self.video_clip_path,
            "extra": self.extra,
        }


# ── Event ─────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    Structured event emitted when a rule is confirmed by the rule engine.

    Lifecycle:
        RuleEngine confirms incident
            → emit Event(state="triggered")
            → NotificationManager consumes Event
            → IncidentRecorder persists Event

    Fields align with Day 2 spec:
        event_id, rule_id, timestamp, camera_id, track_id, evidence, action
    """

    # ── Required fields (always set at creation) ──────────────────────────────
    event_id:   str     # unique ID, format: "evt_{uuid4_short}"
    rule_id:    str     # which rule triggered this event
    camera_id:  str     # source camera identifier (e.g. "cam_01")
    track_id:   int     # ByteTrack track_id of the violating object
    timestamp:  float   # seconds since video start (or wall-clock epoch)

    # ── Action ───────────────────────────────────────────────────────────────
    action:     str     # what should happen: "alert" | "record" | "alert+record"

    # ── Evidence ─────────────────────────────────────────────────────────────
    evidence:   EventEvidence

    # ── Optional / enrichment ────────────────────────────────────────────────
    class_name:     str = "unknown"         # detected class that triggered rule
    rule_description: str = ""             # human-readable rule description
    state:          str = "triggered"      # "triggered" | "resolved" | "suppressed"
    suppressed:     bool = False           # True if within cooldown window
    created_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        rule_id: str,
        camera_id: str,
        track_id: int,
        timestamp: float,
        action: str,
        evidence: EventEvidence,
        class_name: str = "unknown",
        rule_description: str = "",
    ) -> "Event":
        """
        Convenience factory — auto-generates event_id.

        Usage:
            event = Event.create(
                rule_id="no_helmet",
                camera_id="cam_01",
                track_id=7,
                timestamp=12.4,
                action="alert+record",
                evidence=EventEvidence(bbox=[100,50,300,400], confidence=0.82, frame_id=372),
            )
        """
        short_id = str(uuid.uuid4()).replace("-", "")[:10]
        return cls(
            event_id=f"evt_{short_id}",
            rule_id=rule_id,
            camera_id=camera_id,
            track_id=track_id,
            timestamp=timestamp,
            action=action,
            evidence=evidence,
            class_name=class_name,
            rule_description=rule_description,
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "event_id":         self.event_id,
            "rule_id":          self.rule_id,
            "rule_description": self.rule_description,
            "camera_id":        self.camera_id,
            "track_id":         self.track_id,
            "timestamp":        round(self.timestamp, 3),
            "action":           self.action,
            "class_name":       self.class_name,
            "state":            self.state,
            "suppressed":       self.suppressed,
            "evidence":         self.evidence.to_dict(),
            "created_at":       self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return (
            f"[Event {self.event_id}] rule={self.rule_id} "
            f"cam={self.camera_id} track={self.track_id} "
            f"t={self.timestamp:.2f}s action={self.action}"
        )
