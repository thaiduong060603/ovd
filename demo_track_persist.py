"""
demo_track_persist.py
─────────────────────
Day 2 — Task 6: Confirm track_id persists across frames.

Runs the full pipeline:
    detector → ByteTracker → RuleEngineV1 → Event output

For each frame prints a compact table of active tracks so you can
visually verify that the same object keeps the same track_id over time.

Usage:
    # With a real video
    python demo_track_persist.py --input video.mp4 --prompt "person"

    # With a real video + rule file (to test full Event emission)
    python demo_track_persist.py --input video.mp4 --prompt "person" \
        --rule configs/no_helmet.json --max-frames 300

    # CPU-only (no GPU)
    python demo_track_persist.py --input video.mp4 --prompt "person" --device cpu

Output example:
    [f=0030 t=1.00s]  Active tracks: 2
      ID=1  class=person  conf=0.81  state=confirmed  dur=1.00s  bbox=[112,45,298,410]
      ID=2  class=person  conf=0.74  state=tentative  dur=0.23s  bbox=[450,90,610,390]
    ...
    [f=0060 t=2.00s]  Active tracks: 2
      ID=1  class=person  conf=0.79  state=confirmed  dur=2.00s  bbox=[120,47,305,415]  ← same ID ✓
      ID=2  class=person  conf=0.71  state=confirmed  dur=1.23s  bbox=[460,88,618,392]  ← same ID ✓
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Track-ID persistence demo — Day 2")
    p.add_argument("--input",       required=True,        help="Video file or RTSP URL")
    p.add_argument("--prompt",      default="person",     help="Detection prompt (e.g. 'person, forklift')")
    p.add_argument("--detector",    default="yolo_world", choices=["yolo_world", "groundingdino"])
    p.add_argument("--model",       default=None,         help="Model path override")
    p.add_argument("--device",      default="auto",       choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--det-interval",type=int, default=5,  help="Detect every N frames (default: 5)")
    p.add_argument("--rule",        default=None,         help="Optional rule JSON to test Event emission")
    p.add_argument("--camera-id",   default="cam_01",     help="Camera identifier (default: cam_01)")
    p.add_argument("--max-frames",  type=int, default=0,  help="Stop after N frames (0=all)")
    p.add_argument("--display",     action="store_true",  help="Show annotated window")
    p.add_argument("--output",      default=None,         help="Save annotated video")
    return p.parse_args()


# ── device resolver ───────────────────────────────────────────────────────────

def resolve_device(d: str) -> str:
    import torch
    if d != "auto":
        return d
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU: {name} → cuda")
        return "cuda"
    if getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon → mps")
        return "mps"
    print("  ⚠  No GPU → cpu")
    return "cpu"


# ── visualization ─────────────────────────────────────────────────────────────

STATE_COLORS = {
    "confirmed":  (0, 255, 0),
    "tentative":  (0, 165, 255),
    "helmetless": (0, 0, 255),
    "helmeted":   (0, 255, 0),
    "unknown":    (128, 128, 128),
    "lost":       (64, 64, 64),
}

def draw_tracks(frame: np.ndarray, tracks, events_this_frame) -> np.ndarray:
    vis = frame.copy()
    event_track_ids = {e.track_id for e in events_this_frame}

    for track in tracks:
        if track.state == "lost":
            continue
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        color = STATE_COLORS.get(track.state, (200, 200, 200))

        # Thicker border if event fired this frame
        thickness = 3 if track.track_id in event_track_ids else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"ID={track.track_id} {track.class_name[:10]} {track.confidence:.2f}"
        dur   = f"{track.duration:.1f}s [{track.state}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(vis, (x1, y1 - th - 18), (x1 + max(tw, 120) + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 1)
        cv2.putText(vis, dur,   (x1 + 2, y1 - 1),  cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1)

        # Flash "EVENT" badge
        if track.track_id in event_track_ids:
            cv2.putText(vis, "EVENT!", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis


def draw_info(vis, frame_id, fps, n_det, n_tracks, n_events_total, prompt):
    lines = [
        f"Frame : {frame_id}",
        f"FPS   : {fps:.1f}",
        f"Det   : {n_det}",
        f"Tracks: {n_tracks}",
        f"Events: {n_events_total}",
        f"Prompt: {prompt[:30]}",
    ]
    for i, line in enumerate(lines):
        y = 20 + i * 22
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return vis


# ── track-persist table (printed to stdout) ───────────────────────────────────

def print_track_table(frame_id: int, timestamp: float, tracks, new_events):
    active = [t for t in tracks if t.state != "lost"]
    print(f"\n[f={frame_id:05d} t={timestamp:.2f}s]  Active tracks: {len(active)}")
    for t in active:
        event_marker = " ← 📤 EVENT" if any(e.track_id == t.track_id for e in new_events) else ""
        print(
            f"  ID={t.track_id:<4} class={t.class_name:<22} conf={t.confidence:.2f}"
            f"  state={t.state:<12} dur={t.duration:.2f}s"
            f"  bbox=[{int(t.bbox[0])},{int(t.bbox[1])},{int(t.bbox[2])},{int(t.bbox[3])}]"
            f"{event_marker}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.device = resolve_device(args.device)

    print("=" * 65)
    print("  TRACK-ID PERSISTENCE DEMO  (Day 2)")
    print("=" * 65)
    print(f"  Input     : {args.input}")
    print(f"  Prompt    : {args.prompt}")
    print(f"  Detector  : {args.detector}")
    print(f"  Device    : {args.device}")
    print(f"  Camera    : {args.camera_id}")
    print(f"  Det every : {args.det_interval} frames")
    print("=" * 65)

    # ── 1. Detector ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading detector...")
    if args.detector == "yolo_world":
        from src.core.detect.yolo_world_detector import YOLOWorldDetector
        mp = args.model or "models/yolov8s-world.pt"
        detector = YOLOWorldDetector(
            model_path=mp, box_threshold=0.30,
            text_threshold=0.25, device=args.device,
        )
        if args.device != "cpu":
            detector.warmup()
    else:
        from src.core.detect.grounding_dino_detector import GroundingDINODetector
        detector = GroundingDINODetector(
            box_threshold=0.30, text_threshold=0.25, device=args.device,
        )

    # ── 2. Tracker ────────────────────────────────────────────────────────────
    print("\n[2/4] Loading ByteTracker...")
    from src.core.track.byte_tracker import ByteTracker
    # fps resolved after video open — use placeholder 30, updated below
    tracker = ByteTracker(
        track_thresh=0.4, track_buffer=90,
        match_thresh=0.5, frame_rate=30.0,
    )

    # ── 3. Rule engine ────────────────────────────────────────────────────────
    print("\n[3/4] Loading RuleEngineV1...")
    from src.core.rules.rule_engine_core_v1 import RuleEngineV1

    rules = []
    if args.rule:
        from src.models.rule import Rule
        from src.utils.rule_validator import validate_rule_json, parse_rule_from_json
        rpath = Path(args.rule)
        if rpath.suffix.lower() == ".json":
            with open(rpath) as f:
                rdata = json.load(f)
            valid, err = validate_rule_json(rdata)
            if not valid:
                print(f"  ⚠  Invalid rule: {err} — running without rules")
            else:
                rules.append(parse_rule_from_json(rdata))
        elif rpath.suffix.lower() in {".yaml", ".yml"}:
            rules.append(Rule.from_yaml(str(rpath)))
        print(f"  ✓ Rule loaded: {rules[0].rule_id if rules else 'none'}")
    else:
        print("  ℹ  No rule file — tracking only (no incident/event evaluation)")

    rule_engine = RuleEngineV1(rules=rules, camera_id=args.camera_id)

    # ── 4. Video source ───────────────────────────────────────────────────────
    print("\n[4/4] Opening video...")
    from src.core.ingest.video_source import create_video_source
    video_source = create_video_source(args.input)
    fps_src = video_source.fps or 25.0
    tracker.frame_rate = fps_src          # update tracker with real fps
    print(f"  ✓ FPS: {fps_src:.1f}")

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if args.output:
        w = int(video_source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
            print(f"  ✓ Output → {args.output}")

    if args.display:
        try:
            cv2.namedWindow("Track Persist Demo", cv2.WINDOW_NORMAL)
            print("  ✓ Display window ready")
        except cv2.error:
            print("  ⚠  GUI not available — disabling --display")
            args.display = False

    # ── Main loop ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RUNNING — press q to stop")
    print("=" * 65)

    frame_id        = 0
    detection_cache = []
    last_det_frame  = -999
    fps_t0          = time.time()
    fps_count       = 0
    current_fps     = 0.0
    prompts         = [args.prompt]

    # Track ID stability bookkeeping (for final report)
    # id_frame_log: track_id → list of frame_ids seen
    id_frame_log: dict = {}

    try:
        while True:
            ok, frame = video_source.read()
            if not ok:
                break

            frame_id += 1
            timestamp = frame_id / fps_src

            if args.max_frames and frame_id > args.max_frames:
                print(f"\n  ✓ --max-frames={args.max_frames} reached, stopping.")
                break

            # ── DETECT ────────────────────────────────────────────────────────
            if frame_id % args.det_interval == 0:
                detection_cache = detector.detect(frame, prompts, frame_id)
                last_det_frame  = frame_id
                # breakpoint()

            # ── TRACK ─────────────────────────────────────────────────────────
            stale = frame_id - last_det_frame
            det_input = detection_cache if stale < args.det_interval else []
            tracks = tracker.update(det_input, frame_id, timestamp)

            # Record which frames each track_id appears in
            for t in tracks:
                if t.state != "lost":
                    id_frame_log.setdefault(t.track_id, []).append(frame_id)

            # ── RULE ENGINE ───────────────────────────────────────────────────
            new_incidents = []
            new_events    = []
            if rules:
                new_incidents = rule_engine.evaluate(tracks, frame_id, timestamp)
                # Collect events emitted THIS evaluate() call
                new_events = [
                    e for e in rule_engine.events
                    if abs(e.timestamp - timestamp) < (1.0 / fps_src) + 0.01
                ]

            # ── PRINT TABLE (every 30 frames OR when track count changes) ─────
            if frame_id % 30 == 0 or new_events:
                print_track_table(frame_id, timestamp, tracks, new_events)

            if new_events:
                for ev in new_events:
                    print(f"\n  {'─'*55}")
                    print(f"  {ev}")
                    print(f"  {ev.to_json()}")
                    print(f"  {'─'*55}")

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            if fps_count >= 30:
                current_fps = fps_count / (time.time() - fps_t0)
                fps_t0  = time.time()
                fps_count = 0

            # ── VISUALIZE ─────────────────────────────────────────────────────
            if args.display or writer:
                vis = draw_tracks(frame, tracks, new_events)
                vis = draw_info(vis, frame_id, current_fps,
                                len(detection_cache), len(tracks),
                                len(rule_engine.events), args.prompt)
                if writer:
                    writer.write(vis)
                if args.display:
                    cv2.imshow("Track Persist Demo", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    elif k == ord("p"):
                        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\n  Stopped (Ctrl+C)")
    finally:
        video_source.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRACK-ID PERSISTENCE REPORT")
    print("=" * 65)
    print(f"  Frames processed : {frame_id}")
    print(f"  Unique track IDs : {len(id_frame_log)}")
    print(f"  Total events     : {len(rule_engine.events)}")
    print()
    print(f"  {'ID':<6} {'First frame':>12} {'Last frame':>12} {'Span (frames)':>14} {'Span (s)':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*10}")
    for tid, frames in sorted(id_frame_log.items()):
        first, last = frames[0], frames[-1]
        span = last - first + 1
        span_s = span / fps_src
        print(f"  {tid:<6} {first:>12} {last:>12} {span:>14} {span_s:>9.2f}s")

    if rule_engine.events:
        print(f"\n  Events emitted:")
        for ev in rule_engine.events:
            print(f"    {ev}")

    print("=" * 65)


if __name__ == "__main__":
    main()
