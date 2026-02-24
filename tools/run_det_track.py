
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import cv2
import json
import numpy as np


from src.core.ingest.video_source import create_video_source, USBCameraSource
from src.core.detect.grounding_dino_detector import GroundingDINODetector
from src.core.track.byte_tracker import ByteTracker

class SimpleVisualizer:
    """Simple visualizer for detection and tracking"""
    
    def __init__(self):
        self.colors = {}
        
    def get_color(self, track_id):
        """Get consistent color for track ID"""
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]
    
    def draw(self, frame, tracks, detections=None):
        """Draw tracks and detections on frame"""
        vis = frame.copy()
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color = self.get_color(track.track_id)
            
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track.track_id} [{track.state}] {track.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info
        info_text = [
            f"Tracks: {len(tracks)}",
            f"Detections: {len(detections) if detections else 0}",
            f"Frame: {int(time.time() * 30) % 10000}"
        ]
        
        y = 30
        for text in info_text:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            y += 30
        
        return vis


def main():
    parser = argparse.ArgumentParser(
        description="Detection + Tracking Verification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with video file (CPU)
  python tools/run_det_track.py --input assets/sample.mp4 --prompt "person" --device cpu
  
  # Test with USB camera (GPU)
  python tools/run_det_track.py --input 0 --prompt "person vehicle" --device cuda
  
  # Export results
  python tools/run_det_track.py --input video.mp4 --prompt "person" --output outputs/result.mp4 --json
        """
    )
    
    # Input/Output
    parser.add_argument("--input", required=True,
                       help="Video file path or camera index (0, 1, etc.)")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--json", action="store_true",
                       help="Export tracking data to JSON")
    
    # Detection parameters
    parser.add_argument("--prompt", default="person",
                       help="Detection prompt(s), space-separated (default: person)")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                       help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                       help="Text matching threshold (default: 0.25)")
    
    # Tracking parameters
    parser.add_argument("--track-thresh", type=float, default=0.5,
                       help="Track confidence threshold (default: 0.5)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="Track buffer frames (default: 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                       help="IOU matching threshold (default: 0.8)")
    
    # System parameters
    parser.add_argument("--device", default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run on (default: cuda)")
    parser.add_argument("--detection-interval", type=int, default=5,
                       help="Detect every N frames (default: 5)")
    parser.add_argument("--display", action="store_true", default=True,
                       help="Display visualization window")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (default: all)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 DETECTION + TRACKING VERIFICATION")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print(f"Detection interval: {args.detection_interval} frames")
    print("=" * 70)
    
    # Parse prompts
    prompts = args.prompt.split()
    
    # Initialize video source
    print("\n[1/3] Initializing video source...")
    try:
        if args.input.isdigit():
            # USB camera
            camera_id = int(args.input)
            video_source = USBCameraSource(camera_id)
            print(f"✓ USB Camera {camera_id} opened")
        else:
            # Video file
            video_source = create_video_source(args.input)
            print(f"✓ Video file opened: {args.input}")
    except Exception as e:
        print(f"✗ Failed to open input: {e}")
        return
    
    # Initialize detector
    print("\n[2/3] Loading detector...")
    try:
        detector = GroundingDINODetector(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device
        )
        print(f"✓ GroundingDINO loaded on {args.device}")
    except Exception as e:
        print(f"✗ Failed to load detector: {e}")
        if args.device == "cuda":
            print("💡 Try --device cpu")
        return
    
    # Initialize tracker
    print("\n[3/3] Initializing tracker...")
    tracker = ByteTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        frame_rate=video_source.fps
    )
    print("✓ ByteTrack initialized")
    
    # Initialize visualizer
    visualizer = SimpleVisualizer()
    
    # Video writer
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        success, frame = video_source.read()
        if success:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, video_source.fps, (w, h))
            print(f"✓ Output video: {args.output}")
            # Reset source
            video_source.release()
            if args.input.isdigit():
                video_source = USBCameraSource(int(args.input))
            else:
                video_source = create_video_source(args.input)
    
    # JSON export
    json_data = [] if args.json else None
    
    print("\n" + "=" * 70)
    print("▶️  PROCESSING STARTED (Press 'q' to quit, 'p' to pause)")
    print("=" * 70 + "\n")
    
    # Main loop
    frame_id = 0
    detection_cache = []
    last_detection_frame = -999
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0
    
    try:
        while True:
            success, frame = video_source.read()
            if not success:
                break
            
            frame_id += 1
            timestamp = frame_id / video_source.fps
            
            # Check max frames
            if args.max_frames and frame_id > args.max_frames:
                break
            
            # Detection
            should_detect = (frame_id % args.detection_interval == 0)
            
            if should_detect:
                detections = detector.detect(frame, prompts, frame_id)
                detection_cache = detections
                last_detection_frame = frame_id
                
                if frame_id % 30 == 0 or len(detections) > 0:
                    print(f"Frame {frame_id}: Detected {len(detections)} objects")
            
            # Tracking
            frames_since_detection = frame_id - last_detection_frame
            if detection_cache and frames_since_detection < args.detection_interval:
                current_detections = detection_cache
            else:
                current_detections = []
            
            tracks = tracker.update(current_detections, frame_id, timestamp)
            
            # Export to JSON
            if json_data is not None:
                frame_data = {
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'tracks': [
                        {
                            'track_id': t.track_id,
                            'bbox': t.bbox.tolist(),
                            'confidence': float(t.confidence),
                            'class': t.class_name,
                            'state': t.state
                        }
                        for t in tracks
                    ]
                }
                json_data.append(frame_data)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_start = time.time()
                fps_counter = 0
            
            # Visualization
            if args.display or writer:
                vis_frame = visualizer.draw(frame, tracks, current_detections)
                
                # FPS overlay
                cv2.putText(vis_frame, f"FPS: {current_fps:.1f}",
                           (vis_frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if args.display:
                    cv2.imshow("Detection + Tracking", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)
                
                if writer:
                    writer.write(vis_frame)
            
            # Progress
            if frame_id % 100 == 0:
                print(f"Processed {frame_id} frames | FPS: {current_fps:.1f} | Tracks: {len(tracks)}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE")
        print("=" * 70)
        
        video_source.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\nSummary:")
        print(f"  Frames processed: {frame_id}")
        print(f"  Total tracks created: {tracker.next_track_id - 1}")
        print(f"  Average FPS: {current_fps:.1f}")
        
        # Save JSON
        if json_data:
            json_path = args.output.replace('.mp4', '.json') if args.output else 'outputs/tracks.json'
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"  ✓ JSON exported: {json_path}")
        
        if args.output:
            print(f"  ✓ Video saved: {args.output}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()