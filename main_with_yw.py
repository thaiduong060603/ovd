"""
Main pipeline: OVD detection → tracking → rule evaluation → incident management.
"""
import argparse
import time
import cv2
import json
from pathlib import Path

from src.core.ingest.video_source import create_video_source
from src.core.detect.grounding_dino_detector import GroundingDINODetector
from src.core.detect.yolo_world_detector import YOLOWorldDetector
from src.core.track.byte_tracker import ByteTracker
from src.core.rules.rule_engine import RuleEngine
from src.core.record.ring_buffer import VideoRingBuffer
from src.core.record.incident_recorder import IncidentRecorder
from src.core.notify.notification_manager import NotificationManager
from src.models.rule import Rule
from src.utils.visualization import Visualizer
from src.utils.rule_validator import validate_rule_json, parse_rule_from_json


def _check_gui_available() -> bool:
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


def _safe_imshow(window: str, frame) -> str:
    try:
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else ""
    except cv2.error:
        return ""


def _safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def main():
    parser = argparse.ArgumentParser(description="OVD Watchdog — Full Pipeline")
    parser.add_argument("--input",              required=True,       help="Video file / RTSP URL")
    parser.add_argument("--rule",               required=True,       help="Rule DSL file (.json) or legacy (.yaml)")
    parser.add_argument("--detection-interval", type=int, default=5, help="Detect every N frames")
    parser.add_argument("--display",            action="store_true", help="Show visualization window")
    parser.add_argument("--output",             default=None,        help="Output video path (optional)")
    parser.add_argument("--detector",           default="yolo_world",
                        choices=["yolo_world", "groundingdino"],
                        help="Detector backend (default: yolo_world)")
    parser.add_argument("--model",              default=None,
                        help="Path to model file. Default: models/yolov8s-world.pt (yolo_world) "
                             "or models/groundingdino_swint_ogc.pth (groundingdino)")
    parser.add_argument("--device",             default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"],
                        help="Device to run inference on (default: auto = use GPU if available)")
    args = parser.parse_args()

    args.device = _resolve_device(args.device)
    _print_header(args)

    # ── 1. Load rule ──────────────────────────────────────────
    print("\n[1/8] Loading rule...")
    rule = _load_rule(args.rule)
    print(f"  ✓ [{rule.rule_id}] {rule.description}")
    print(f"    method    : {getattr(rule, 'method', 'composite')}")
    print(f"    dwell     : {rule.conditions.dwell_seconds}s")
    print(f"    ROI       : {rule.roi.enabled if rule.roi else False}")

    # ── 2. Video source ───────────────────────────────────────
    print("\n[2/8] Opening video source...")
    video_source = create_video_source(args.input)

    # ── 3. Detector ───────────────────────────────────────────
    detector = _create_detector(args, rule)

    # ── 4. Tracker ────────────────────────────────────────────
    print("\n[4/8] Initializing ByteTracker...")
    tracker = ByteTracker(
        track_thresh=0.4,
        track_buffer=90,
        match_thresh=0.5,
        frame_rate=video_source.fps,
    )

    # ── 5. Rule engine ────────────────────────────────────────
    print("\n[5/8] Initializing RuleEngine...")
    rule_engine = RuleEngine([rule])

    # ── 6. Ring buffer + recorder ─────────────────────────────
    print("\n[6/8] Initializing recording system...")
    ring_buffer = VideoRingBuffer(
        max_seconds=int(rule.actions.record_pre_seconds + rule.actions.record_post_seconds + 10),
        fps=video_source.fps,
    )
    incident_recorder = IncidentRecorder(ring_buffer)

    # ── 7. Notifications ──────────────────────────────────────
    print("\n[7/8] Initializing notifications...")
    notification_manager = NotificationManager()

    # ── 8. Visualizer + GUI check + writer ───────────────────
    print("\n[8/8] Starting pipeline...")
    visualizer = Visualizer()

    if args.display:
        if _check_gui_available():
            print("  ✓ GUI available")
        else:
            print("  ⚠  --display requested but OpenCV GUI is not available")
            print("     (Likely opencv-python-headless is installed)")
            print("     Fix: pip uninstall opencv-python-headless && pip install opencv-python")
            print("     Continuing without display — output file will still be saved.")
            args.display = False

    writer = None
    if args.output:
        w = int(video_source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, video_source.fps, (w, h))
            print(f"  ✓ Recording output → {args.output}  ({w}x{h})")
        else:
            print("  ⚠  Could not determine frame size — output video disabled")

    # ── Main loop ─────────────────────────────────────────────
    frame_id        = 0
    detection_cache = []
    last_det_frame  = -999
    fps_start       = time.time()
    fps_counter     = 0
    current_fps     = 0.0
    recorded_ids: set = set()
    stop_requested  = False

    _print_separator("MONITORING STARTED")

    try:
        while True:
            ok, frame = video_source.read()
            if not ok:
                break

            frame_id += 1
            timestamp = frame_id / video_source.fps

            ring_buffer.add_frame(frame, timestamp, frame_id)

            # DETECTION (sparse)
            if frame_id % args.detection_interval == 0:
                print(f"\n[DETECT] frame={frame_id}  t={timestamp:.2f}s")
                detection_cache = detector.detect(frame, [rule.prompt_positive], frame_id)
                last_det_frame  = frame_id
                n = len(detection_cache)
                print(f"         → {n} object{'s' if n != 1 else ''} detected")

            # TRACKING (dense)
            frames_stale = frame_id - last_det_frame
            det_input = detection_cache if frames_stale < args.detection_interval else []
            tracks = tracker.update(det_input, frame_id, timestamp)

            if frame_id % 30 == 0:
                ids = [t.track_id for t in tracks[:5]]
                print(f"[TRACK]  frame={frame_id}: {len(det_input)} det → {len(tracks)} tracks  IDs={ids}")

            # RULE EVALUATION
            incidents = rule_engine.evaluate(tracks, frame_id, timestamp)

            # INCIDENT HANDLING
            for incident in incidents:
                if incident.state == "confirmed" and incident.incident_id not in recorded_ids:
                    if rule_engine.should_notify(incident, timestamp, rule):
                        _print_separator(f"INCIDENT: {incident.incident_id}")

                        incident_recorder.record_incident(incident, rule, timestamp)

                        try:
                            results = notification_manager.notify(incident, rule)
                            for ch, ok_n in results.items():
                                print(f"  {'✓' if ok_n else '✗'} {ch}")
                        except Exception as e:
                            print(f"  ✗ notification error: {e}")

                        rule_engine.mark_notified(incident, timestamp)
                        recorded_ids.add(incident.incident_id)
                        _print_separator()

            # FPS
            fps_counter += 1
            if fps_counter >= 30:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_start   = time.time()
                fps_counter = 0

            # VISUALIZATION
            if args.display or writer:
                vis = frame.copy()
                if rule.roi and rule.roi.enabled:
                    vis = visualizer.draw_roi(vis, rule.roi)
                vis = visualizer.draw_tracks(vis, tracks, show_id=True, show_state=True)
                vis = visualizer.draw_incidents(vis, rule_engine.get_active_incidents(), tracks)
                vis = visualizer.add_info_panel_with_incidents(
                    vis, frame_id, current_fps,
                    len(detection_cache), len(tracks),
                    len(rule_engine.get_active_incidents()),
                    len(rule_engine.get_confirmed_incidents()),
                )
                if frame_id % args.detection_interval == 0:
                    cv2.putText(vis, "DETECTING", (vis.shape[1] - 200, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if args.display:
                    key = _safe_imshow("OVD Watchdog", vis)
                    if key == "q":
                        stop_requested = True
                    elif key == "p":
                        try:
                            cv2.waitKey(0) 
                        except cv2.error:
                            pass

                if writer:
                    writer.write(vis)

            if frame_id % 100 == 0:
                print(
                    f"[PROGRESS] frame={frame_id}  fps={current_fps:.1f}"
                    f"  tracks={len(tracks)}  incidents={len(rule_engine.get_active_incidents())}"
                    f"  confirmed={len(rule_engine.get_confirmed_incidents())}"
                )

            if stop_requested:
                print("\n⚠  Stopped by user (q)")
                break

    except KeyboardInterrupt:
        print("\n\n⚠  Stopped by user (Ctrl+C)")

    finally:
        video_source.release()
        if writer:
            writer.release()
        _safe_destroy_windows()
        _print_summary(frame_id, tracker, rule_engine, recorded_ids)


# ── Helpers ───────────────────────────────────────────────────

def _create_detector(args, rule):
    """Khởi tạo detector đúng loại theo --detector argument."""
    if args.detector == "yolo_world":
        model_path = args.model or "models/yolov8s-world.pt"
        print(f"\n[3/8] Loading YOLO-World ({model_path})...")
        detector = YOLOWorldDetector(
            model_path=model_path,
            box_threshold=rule.box_threshold,
            text_threshold=rule.text_threshold,
            device=args.device,
        )
        if args.device != "cpu":
            detector.warmup()
        return detector

    elif args.detector == "groundingdino":
        model_path = args.model or "models/groundingdino_swint_ogc.pth"
        print(f"\n[3/8] Loading GroundingDINO ({model_path})...")
        return GroundingDINODetector(
            box_threshold=rule.box_threshold,
            text_threshold=rule.text_threshold,
            device=args.device,
        )

    raise ValueError(f"Unknown detector: {args.detector}")


def _resolve_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU detected: {gpu_name} ({vram:.1f} GB VRAM) → using cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print(f"  ✓ Apple Silicon GPU detected → using mps")
        else:
            device = "cpu"
            print(f"  ⚠  No GPU detected → using cpu")
        return device

    if device_arg in ("cuda", "cuda:0", "cuda:1"):
        if not torch.cuda.is_available():
            msg = (
                f"Device '{device_arg}' requested but CUDA is not available. "
                "Check: nvidia-smi | pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
            raise RuntimeError(msg)
        idx = int(device_arg.split(":")[1]) if ":" in device_arg else 0
        gpu_name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  ✓ Using {device_arg}: {gpu_name} ({vram:.1f} GB VRAM)")
        return device_arg

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Device 'mps' requested but MPS is not available (requires Apple Silicon + macOS 12+)")
        print(f"  ✓ Using mps (Apple Silicon)")
        return "mps"

    # cpu
    print(f"  ✓ Using cpu (forced)")
    return "cpu"


def _load_rule(rule_path_str: str) -> Rule:
    path = Path(rule_path_str)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid, err = validate_rule_json(data)
        if not valid:
            raise ValueError(f"Invalid rule JSON: {err}")
        return parse_rule_from_json(data)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        return Rule.from_yaml(str(path))
    raise ValueError(f"Unsupported rule file format: {path.suffix}")


def _print_header(args):
    print("=" * 70)
    print("  OVD WATCHDOG SYSTEM")
    print("=" * 70)
    print(f"  Input : {args.input}")
    print(f"  Rule  : {args.rule}")
    print(f"  Det.  : every {args.detection_interval} frames")
    print(f"  Detector: {args.detector}")
    print(f"  Device: {args.device}")
    print("=" * 70)


def _print_separator(label: str = ""):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
    else:
        print("=" * 70)


def _print_summary(frame_id: int, tracker, rule_engine, recorded_ids: set):
    _print_separator("SESSION SUMMARY")
    print(f"  Frames processed : {frame_id}")
    print(f"  Tracks created   : {tracker.next_track_id - 1}")
    print(f"  Total incidents  : {len(rule_engine.incidents)}")
    print(f"  Confirmed        : {len(rule_engine.get_confirmed_incidents())}")
    print(f"  Recorded         : {len(recorded_ids)}")

    confirmed = rule_engine.get_confirmed_incidents()
    if confirmed:
        print("\n  Confirmed incidents:")
        for inc in confirmed:
            print(f"    [{inc.incident_id}]")
            print(f"      track={inc.track_id}  time={inc.confirmed_time:.2f}s"
                  f"  notified={inc.notification_count}x")
            print(f"      snapshots={len(inc.snapshots)}  clip={inc.video_clip_path}")

    print(f"\n  Recordings → data/incidents/")
    print("=" * 70)


if __name__ == "__main__":
    main()
