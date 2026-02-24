"""
Main pipeline for OVD detection, tracking, and incident management
"""
import argparse
import time
import cv2
from pathlib import Path

from src.core.ingest.video_source import create_video_source
from src.core.detect.grounding_dino_detector import GroundingDINODetector
from src.core.track.byte_tracker import ByteTracker
from src.core.rules.rule_engine import RuleEngine
from src.core.record.ring_buffer import VideoRingBuffer
from src.core.record.incident_recorder import IncidentRecorder
from src.core.notify.notification_manager import NotificationManager  # NEW
from src.models.rule import Rule
from src.utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="OVD Full Pipeline with Rule Engine")
    parser.add_argument("--input", required=True, help="Video file path or RTSP URL")
    parser.add_argument("--rule", required=True, help="Path to rule YAML file")
    parser.add_argument("--detection-interval", type=int, default=30, help="Detect every N frames")
    parser.add_argument("--display", action="store_true", help="Display visualization")
    parser.add_argument("--output", help="Output video path (optional)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 OVD WATCHDOG SYSTEM - FULL PIPELINE")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Rule: {args.rule}")
    print(f"Detection interval: {args.detection_interval} frames")
    print("=" * 70)
    
    # Load rule
    print("\n[1/8] Loading monitoring rule...")  # Changed 1/7 → 1/8
    rule = Rule.from_yaml(args.rule)
    print(f"✓ Rule loaded: {rule.rule_id}")
    print(f"  Description: {rule.description}")
    print(f"  Dwell time: {rule.conditions.dwell_seconds}s")
    print(f"  ROI enabled: {rule.roi.enabled if rule.roi else False}")
    
    # Initialize video source
    print("\n[2/8] Initializing video source...")
    video_source = create_video_source(args.input)
    
    # Initialize detector
    print("\n[3/8] Loading GroundingDINO detector...")
    detector = GroundingDINODetector(
        box_threshold=rule.box_threshold,
        text_threshold=rule.text_threshold
    )
    
    # Initialize tracker
    print("\n[4/8] Initializing ByteTrack tracker...")
    tracker = ByteTracker(
        track_thresh=0.4,
        track_buffer=90,
        match_thresh=0.5,
        frame_rate=video_source.fps
    )
    
    # Initialize rule engine
    print("\n[5/8] Initializing rule engine...")
    rule_engine = RuleEngine([rule])
    
    # Initialize ring buffer and recorder
    print("\n[6/8] Initializing recording system...")
    ring_buffer = VideoRingBuffer(
        max_seconds=int(rule.actions.record_pre_seconds + rule.actions.record_post_seconds + 10),
        fps=video_source.fps
    )
    incident_recorder = IncidentRecorder(ring_buffer)
    
    # Initialize notification manager (NEW)
    print("\n[7/8] Initializing notification system...")
    notification_manager = NotificationManager()
    
    # Initialize visualization
    print("\n[8/8] Starting pipeline...")
    visualizer = Visualizer()
    
    # Video writer if output specified
    writer = None
    if args.output:
        success, frame = video_source.read()
        if success:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, video_source.fps, (w, h))
            print(f"Saving output to: {args.output}")
            video_source.release()
            video_source = create_video_source(args.input)
    
    # Main loop variables
    frame_id = 0
    detection_cache = []
    last_detection_frame = -999
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0
    recorded_incidents = set()
    
    print("\n" + "=" * 70)
    print("🎬 MONITORING STARTED")
    print("=" * 70)
    
    try:
        while True:
            success, frame = video_source.read()
            if not success:
                break
            
            frame_id += 1
            timestamp = frame_id / video_source.fps
            
            # Add frame to ring buffer
            ring_buffer.add_frame(frame, timestamp, frame_id)
            
            # ============================================================
            # DETECTION - Sparse (every N frames)
            # ============================================================
            should_detect = (frame_id % args.detection_interval == 0)
            
            if should_detect:
                print(f"\n[DETECT] Frame {frame_id} ({timestamp:.2f}s)")
                detections = detector.detect(frame, [rule.prompt_positive], frame_id)
                
                detection_cache = detections
                last_detection_frame = frame_id
                
                if detections:
                    print(f"         → {len(detections)} objects detected")
                else:
                    print(f"         → No objects")
            
            # ============================================================
            # TRACKING - Dense (every frame)
            # ============================================================
            
            frames_since_last_detection = frame_id - last_detection_frame
            
            if detection_cache and frames_since_last_detection < args.detection_interval:
                current_detections = detection_cache
            else:
                current_detections = []
            
            tracks = tracker.update(current_detections, frame_id, timestamp)
            
            if should_detect or frame_id % 30 == 0:
                active_ids = [t.track_id for t in tracks[:5]]
                print(f"[TRACK]  Frame {frame_id}: {len(current_detections)} inputs → {len(tracks)} tracks (IDs: {active_ids})")
            
            # ============================================================
            # RULE EVALUATION (every frame)
            # ============================================================
            incidents = rule_engine.evaluate(tracks, frame_id, timestamp)
            
            # ============================================================
            # INCIDENT HANDLING WITH NOTIFICATIONS (NEW)
            # ============================================================
            for incident in incidents:
                if incident.state == "confirmed" and incident.incident_id not in recorded_incidents:
                    if rule_engine.should_notify(incident, timestamp, rule):
                        print(f"\n{'='*70}")
                        print(f"🚨 INCIDENT CONFIRMED: {incident.incident_id}")
                        print(f"   Track ID: {incident.track_id}")
                        print(f"   Time: {incident.confirmed_time:.2f}s")
                        print(f"{'='*70}")
                        
                        # Record incident evidence
                        print(f"\n[RECORD] Recording incident evidence...")
                        incident_recorder.record_incident(incident, rule, timestamp)
                        
                        # Send notifications (NEW)
                        print(f"\n[NOTIFY] Sending notifications...")
                        try:
                            notification_results = notification_manager.notify(incident, rule)
                            
                            # Print notification results
                            for channel, success in notification_results.items():
                                status = "✓" if success else "✗"
                                print(f"  {status} {channel}: {'Sent' if success else 'Failed'}")
                        except Exception as e:
                            print(f"  ✗ Notification error: {e}")
                        
                        # Mark as notified
                        rule_engine.mark_notified(incident, timestamp)
                        recorded_incidents.add(incident.incident_id)
                        
                        print(f"{'='*70}\n")
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_start = time.time()
                fps_counter = 0
            
            # ============================================================
            # VISUALIZATION
            # ============================================================
            if args.display or writer:
                vis_frame = frame.copy()
                
                # Draw ROI
                if rule.roi and rule.roi.enabled:
                    vis_frame = visualizer.draw_roi(vis_frame, rule.roi)
                
                # Draw tracks
                vis_frame = visualizer.draw_tracks(vis_frame, tracks, show_id=True, show_state=True)
                
                # Draw incidents
                vis_frame = visualizer.draw_incidents(vis_frame, rule_engine.get_active_incidents(), tracks)
                
                # Info panel
                vis_frame = visualizer.add_info_panel_with_incidents(
                    vis_frame,
                    frame_id,
                    current_fps,
                    len(detection_cache),
                    len(tracks),
                    len(rule_engine.get_active_incidents()),
                    len(rule_engine.get_confirmed_incidents())
                )
                
                # Detection indicator
                if should_detect:
                    cv2.putText(
                        vis_frame,
                        "DETECTING...",
                        (vis_frame.shape[1] - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                
                if args.display:
                    cv2.imshow("OVD Watchdog", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)
                
                if writer:
                    writer.write(vis_frame)
            
            # Progress
            if frame_id % 100 == 0:
                active_inc = len(rule_engine.get_active_incidents())
                confirmed_inc = len(rule_engine.get_confirmed_incidents())
                unique_tracks = len(set(t.track_id for t in tracker.tracks.values()))
                print(f"[PROGRESS] Frame {frame_id} | FPS: {current_fps:.1f} | Tracks: {len(tracks)} (unique: {unique_tracks}) | Incidents: {active_inc} ({confirmed_inc} confirmed)")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("🛑 MONITORING STOPPED")
        print("=" * 70)
        
        video_source.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\n📊 SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Total frames processed: {frame_id}")
        print(f"Total tracks created: {tracker.next_track_id - 1}")
        print(f"Total incidents detected: {len(rule_engine.incidents)}")
        print(f"Confirmed incidents: {len(rule_engine.get_confirmed_incidents())}")
        print(f"Recorded incidents: {len(recorded_incidents)}")
        
        # List incidents with notification info
        if rule_engine.get_confirmed_incidents():
            print(f"\n📋 CONFIRMED INCIDENTS:")
            for inc in rule_engine.get_confirmed_incidents():
                print(f"  - {inc.incident_id}")
                print(f"    Track: {inc.track_id}")
                print(f"    Time: {inc.confirmed_time:.2f}s")
                print(f"    Notifications sent: {inc.notification_count}")
                print(f"    Snapshots: {len(inc.snapshots)}")
                print(f"    Video: {inc.video_clip_path}")
        
        print(f"\n✓ Incident recordings saved to: data/incidents/")
        print("=" * 70)


if __name__ == "__main__":
    main()