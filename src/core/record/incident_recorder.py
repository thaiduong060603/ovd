"""
Incident recorder - manages snapshots and clips for incidents
"""
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from src.models.rule import Incident, Rule
from src.core.record.ring_buffer import VideoRingBuffer


class IncidentRecorder:
    """
    Records evidence (snapshots + clips) for incidents
    """
    
    def __init__(
        self,
        ring_buffer: VideoRingBuffer,
        output_dir: str = "data/incidents"
    ):
        """
        Initialize incident recorder
        
        Args:
            ring_buffer: VideoRingBuffer instance
            output_dir: Base directory for incident recordings
        """
        self.ring_buffer = ring_buffer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Incident Recorder: output → {self.output_dir}")
    
    def record_incident(
        self,
        incident: Incident,
        rule: Rule,
        current_timestamp: float
    ) -> bool:
        """
        Record evidence for confirmed incident
        
        Args:
            incident: Incident object
            rule: Associated rule
            current_timestamp: Current timestamp
        
        Returns:
            True if successful
        """
        if incident.state != "confirmed":
            return False
        
        # Create incident directory
        incident_dir = self.output_dir / incident.incident_id
        incident_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📹 Recording incident: {incident.incident_id}")
        
        # 1. Save snapshot at confirmation time
        snapshot_path = incident_dir / "snapshot.jpg"
        saved_snapshot = self.ring_buffer.save_snapshot(
            incident.confirmed_time or current_timestamp,
            str(snapshot_path)
        )
        
        if saved_snapshot:
            incident.snapshots.append(str(snapshot_path))
        
        # 2. Extract video clip
        clip_path = incident_dir / "clip.mp4"
        saved_clip = self.ring_buffer.extract_clip(
            trigger_timestamp=incident.confirmed_time or current_timestamp,
            pre_seconds=rule.actions.record_pre_seconds,
            post_seconds=rule.actions.record_post_seconds,
            output_path=str(clip_path)
        )
        
        if saved_clip:
            incident.video_clip_path = str(clip_path)
        
        # 3. Save metadata JSON
        metadata_path = incident_dir / "metadata.json"
        self.save_metadata(incident, rule, metadata_path)
        
        print(f"✓ Incident recorded to: {incident_dir}")
        
        return True
    
    def save_metadata(
        self,
        incident: Incident,
        rule: Rule,
        output_path: Path
    ):
        """
        Save incident metadata as JSON
        """
        metadata = {
            'incident_id': incident.incident_id,
            'rule_id': rule.rule_id,
            'rule_description': rule.description,
            'track_id': incident.track_id,
            'state': incident.state,
            'timestamps': {
                'first_detected': incident.first_detected_time,
                'confirmed': incident.confirmed_time,
                'resolved': incident.resolved_time
            },
            'evidence': {
                'snapshots': incident.snapshots,
                'video_clip': incident.video_clip_path
            },
            'statistics': {
                'avg_confidence': sum(incident.confidence_scores) / len(incident.confidence_scores) if incident.confidence_scores else 0,
                'num_frames': len(incident.frame_ids),
                'frame_ids': incident.frame_ids[:10]  # First 10 frames
            },
            'rule_config': {
                'dwell_seconds': rule.conditions.dwell_seconds,
                'cooldown_seconds': rule.actions.cooldown_seconds,
                'roi_enabled': rule.roi.enabled if rule.roi else False
            },
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {output_path}")