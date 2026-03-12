"""
benchmark.py  —  Day 3
Measures FPS, inference time, tracker overhead, and memory usage.

Usage:
    python benchmark.py --input video.mp4
    python benchmark.py --input video.mp4 --detector groundingdino --frames 200
    python benchmark.py --input video.mp4 --det-interval 1   # worst case: detect every frame
    python benchmark.py --input video.mp4 --det-interval 10  # sparse mode

Output:
    - Per-component timing breakdown printed to console
    - Summary table saved to benchmark_result.json
"""

import argparse
import time
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("  ⚠  psutil not installed — memory tracking disabled")
    print("     Fix: pip install psutil")


# ── Timer helper ──────────────────────────────────────────────────────────────

class Timer:
    """Simple accumulating timer."""
    def __init__(self):
        self.total   = 0.0
        self.count   = 0
        self.min     = float("inf")
        self.max     = 0.0
        self._start  = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        self.total += elapsed
        self.count += 1
        self.min = min(self.min, elapsed)
        self.max = max(self.max, elapsed)
        self._start = None

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def summary(self) -> dict:
        return {
            "count":    self.count,
            "avg_ms":   round(self.avg * 1000, 2),
            "min_ms":   round(self.min * 1000, 2) if self.count else 0,
            "max_ms":   round(self.max * 1000, 2),
            "total_s":  round(self.total, 3),
        }


# ── Device resolver ───────────────────────────────────────────────────────────

def resolve_device(d: str) -> str:
    import torch
    if d != "auto":
        return d
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ GPU: {name} ({vram:.1f} GB) → cuda")
        return "cuda"
    if getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon → mps")
        return "mps"
    print("  ⚠  No GPU → cpu")
    return "cpu"


# ── Memory snapshot ───────────────────────────────────────────────────────────

def get_memory_mb() -> dict:
    result = {"process_ram_mb": 0.0, "gpu_vram_mb": 0.0}
    if HAS_PSUTIL:
        import psutil, os
        proc = psutil.Process(os.getpid())
        result["process_ram_mb"] = round(proc.memory_info().rss / 1024**2, 1)
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_vram_mb"] = round(
                torch.cuda.memory_allocated() / 1024**2, 1
            )
    except Exception:
        pass
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark — Day 3")
    parser.add_argument("--input",        required=True,         help="Video file")
    parser.add_argument("--prompt",       default="person",      help="Detection prompt")
    parser.add_argument("--detector",     default="yolo_world",  choices=["yolo_world", "groundingdino"])
    parser.add_argument("--model",        default=None)
    parser.add_argument("--device",       default="auto",        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--det-interval", type=int, default=5,   help="Detect every N frames")
    parser.add_argument("--frames",       type=int, default=300, help="Number of frames to benchmark")
    parser.add_argument("--output",       default="benchmark_result.json")
    args = parser.parse_args()

    args.device = resolve_device(args.device)

    print("=" * 60)
    print("  OVD WATCHDOG — BENCHMARK  (Day 3)")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Detector   : {args.detector}")
    print(f"  Device     : {args.device}")
    print(f"  Det every  : {args.det_interval} frames")
    print(f"  Frames     : {args.frames}")
    print("=" * 60)

    # ── Load detector ─────────────────────────────────────────────────────────
    print("\n[1/3] Loading detector...")
    if args.detector == "yolo_world":
        from src.core.detect.yolo_world_detector import YOLOWorldDetector
        detector = YOLOWorldDetector(
            model_path=args.model or "models/yolov8s-world.pt",
            box_threshold=0.30, text_threshold=0.25, device=args.device,
        )
        if args.device != "cpu":
            detector.warmup()
    else:
        from src.core.detect.grounding_dino_detector import GroundingDINODetector
        detector = GroundingDINODetector(
            box_threshold=0.30, text_threshold=0.25, device=args.device,
        )

    # ── Load tracker ──────────────────────────────────────────────────────────
    print("\n[2/3] Loading tracker...")
    from src.core.track.byte_tracker import ByteTracker
    tracker = ByteTracker(track_thresh=0.4, track_buffer=90, match_thresh=0.5, frame_rate=30.0)

    # ── Open video ────────────────────────────────────────────────────────────
    print("\n[3/3] Opening video...")
    from src.core.ingest.video_source import create_video_source
    video_source = create_video_source(args.input)
    fps_src = video_source.fps or 25.0
    print(f"  ✓ Source FPS: {fps_src:.1f}")

    # ── Timers ────────────────────────────────────────────────────────────────
    t_total     = Timer()
    t_read      = Timer()
    t_detect    = Timer()
    t_track     = Timer()
    t_other     = Timer()

    # ── Memory baseline ───────────────────────────────────────────────────────
    mem_baseline = get_memory_mb()
    print(f"\n  Memory baseline: RAM={mem_baseline['process_ram_mb']} MB  "
          f"VRAM={mem_baseline['gpu_vram_mb']} MB")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    print(f"\n  Running {args.frames} frames...")
    print("  " + "-" * 56)

    frame_id        = 0
    detection_cache = []
    last_det_frame  = -999
    total_detections = 0
    detection_frames = 0

    while frame_id < args.frames:
        t_total.start()

        # READ
        t_read.start()
        ok, frame = video_source.read()
        t_read.stop()
        if not ok:
            break

        frame_id += 1
        timestamp = frame_id / fps_src

        # DETECT
        if frame_id % args.det_interval == 0:
            t_detect.start()
            detection_cache = detector.detect(frame, [args.prompt], frame_id)
            last_det_frame  = frame_id
            t_detect.stop()
            total_detections += len(detection_cache)
            detection_frames += 1

        # TRACK
        stale     = frame_id - last_det_frame
        det_input = detection_cache if stale < args.det_interval else []
        t_track.start()
        tracks = tracker.update(det_input, frame_id, timestamp)
        t_track.stop()

        # OTHER (rule eval placeholder — not timed separately here)
        t_other.start()
        t_other.stop()

        t_total.stop()

        # Progress print every 50 frames
        if frame_id % 50 == 0:
            pipeline_fps = 1.0 / t_total.avg if t_total.avg > 0 else 0
            print(f"  frame={frame_id:04d}  "
                  f"pipeline_fps={pipeline_fps:.1f}  "
                  f"det={t_detect.avg*1000:.1f}ms  "
                  f"track={t_track.avg*1000:.1f}ms  "
                  f"tracks={len(tracks)}")

    video_source.release()

    # ── Memory after ─────────────────────────────────────────────────────────
    mem_after = get_memory_mb()

    # ── Results ───────────────────────────────────────────────────────────────
    pipeline_fps   = 1.0 / t_total.avg if t_total.avg > 0 else 0
    detect_ratio   = detection_frames / max(frame_id, 1)
    avg_det_per_frame = total_detections / max(detection_frames, 1)

    result = {
        "config": {
            "detector":     args.detector,
            "device":       args.device,
            "det_interval": args.det_interval,
            "frames":       frame_id,
            "prompt":       args.prompt,
        },
        "performance": {
            "pipeline_fps":          round(pipeline_fps, 2),
            "frame_read":            t_read.summary(),
            "detection":             t_detect.summary(),
            "tracking":              t_track.summary(),
            "full_frame":            t_total.summary(),
        },
        "detection_stats": {
            "detection_frames":      detection_frames,
            "detection_ratio_pct":   round(detect_ratio * 100, 1),
            "total_detections":      total_detections,
            "avg_det_per_det_frame": round(avg_det_per_frame, 2),
        },
        "memory": {
            "baseline_ram_mb":  mem_baseline["process_ram_mb"],
            "after_ram_mb":     mem_after["process_ram_mb"],
            "delta_ram_mb":     round(mem_after["process_ram_mb"] - mem_baseline["process_ram_mb"], 1),
            "baseline_vram_mb": mem_baseline["gpu_vram_mb"],
            "after_vram_mb":    mem_after["gpu_vram_mb"],
        }
    }

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Frames processed   : {frame_id}")
    print(f"  Pipeline FPS       : {pipeline_fps:.2f}")
    print(f"  ")
    print(f"  Component timing (avg per call):")
    print(f"    Frame read       : {t_read.avg*1000:.2f} ms")
    print(f"    Detection        : {t_detect.avg*1000:.2f} ms  "
          f"(every {args.det_interval} frames → "
          f"effective {t_detect.avg*1000/args.det_interval:.2f} ms/frame)")
    print(f"    Tracking         : {t_track.avg*1000:.2f} ms")
    print(f"    Total per frame  : {t_total.avg*1000:.2f} ms")
    print(f"  ")
    print(f"  Detection stats:")
    print(f"    Detect frames    : {detection_frames} ({detect_ratio*100:.1f}% of total)")
    print(f"    Avg detections   : {avg_det_per_frame:.2f} per detect frame")
    print(f"  ")
    print(f"  Memory:")
    print(f"    RAM delta        : +{result['memory']['delta_ram_mb']} MB")
    print(f"    VRAM after       : {result['memory']['after_vram_mb']} MB")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Results saved → {out}")


if __name__ == "__main__":
    main()
