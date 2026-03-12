"""
inference_server.py  —  OVD Watchdog WebSocket Inference Server
Chạy trên Jetson. Laptop kết nối vào để xem live feed + nhận events.

Install (Jetson):
    pip install fastapi uvicorn[standard] opencv-python pyyaml websockets

Run:
    python inference_server.py --input video.mp4 --rule configs/rules/polygon_intrusion_fullscreen_test.yaml
    python inference_server.py --input video.mp4 --rule configs/rules/dwell_in_zone.yaml --port 8765

Laptop connect:
    python inference_client.py --host <tailscale-ip-jetson> --port 8765

TODO:
    - [ ] Camera input (replace VideoFileSource with cv2.VideoCapture(0))
    - [ ] Accept video file uploaded from laptop via HTTP endpoint
    - [ ] RTSP stream input
"""

import argparse
import asyncio
import base64
import json
import time
import threading
import queue
from pathlib import Path
from typing import Set

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="OVD Watchdog Inference Server")

# Shared state between pipeline thread and WebSocket handlers
_frame_queue: queue.Queue = queue.Queue(maxsize=4)   # (jpeg_bytes, meta_dict)
_event_queue: queue.Queue = queue.Queue(maxsize=100) # event dicts
_pipeline_status = {
    "state":     "idle",     # idle | running | stopped | error
    "fps":       0.0,
    "tracks":    0,
    "incidents": 0,
    "events":    0,
    "frame_id":  0,
    "error":     "",
}
_stop_event   = threading.Event()
_active_ws:   Set[WebSocket] = set()
_pipeline_args = {}


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    """Quick health check — call from laptop to verify server is up."""
    return JSONResponse({
        "server": "ovd-watchdog",
        "status": _pipeline_status,
    })


@app.get("/events")
def get_recent_events():
    """Drain and return all buffered events as JSON list."""
    events = []
    try:
        while True:
            events.append(_event_queue.get_nowait())
    except queue.Empty:
        pass
    return JSONResponse({"events": events})


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _active_ws.add(ws)
    print(f"[WS] Client connected: {ws.client}")
    try:
        # Send current status immediately on connect
        await ws.send_text(json.dumps({
            "type":   "status",
            "payload": _pipeline_status
        }))

        while True:
            # Non-blocking check for new frame
            try:
                jpeg_bytes, meta = _frame_queue.get_nowait()
                # Encode frame as base64 to send over JSON
                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                msg = json.dumps({
                    "type":    "frame",
                    "payload": {
                        "image":   b64,         # base64 JPEG
                        "meta":    meta,         # fps, tracks, incidents, etc.
                    }
                })
                await ws.send_text(msg)
            except queue.Empty:
                pass

            # Drain event queue and send each event
            try:
                while True:
                    evt = _event_queue.get_nowait()
                    await ws.send_text(json.dumps({
                        "type":    "event",
                        "payload": evt,
                    }))
            except queue.Empty:
                pass

            # Small sleep so we don't busy-loop
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {ws.client}")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        _active_ws.discard(ws)


# ── Pipeline thread ────────────────────────────────────────────────────────────

def run_pipeline(args):
    """
    Full OVD pipeline runs in a background thread.
    Pushes annotated JPEG frames and Events into shared queues
    for WebSocket clients to consume.
    """
    global _pipeline_status

    try:
        import sys, os
        sys.path.insert(0, os.getcwd())

        from src.core.ingest.video_source import create_video_source
        from src.core.track.byte_tracker import ByteTracker
        from src.core.rules.rule_engine_core_v1 import RuleEngineV1
        from src.models.rule import Rule

        # ── Load rule ─────────────────────────────────────────────────────────
        print(f"[PIPELINE] Loading rule: {args.rule}")
        rule = Rule.from_yaml(args.rule)
        print(f"[PIPELINE] Rule loaded: [{rule.rule_id}] {rule.description}")

        # ── Video source ──────────────────────────────────────────────────────
        print(f"[PIPELINE] Opening video: {args.input}")
        video_src = create_video_source(args.input)
        fps_src   = video_src.fps or 25.0
        print(f"[PIPELINE] Source FPS: {fps_src:.1f}")

        # ── Detector ──────────────────────────────────────────────────────────
        if args.detector == "yolo_world":
            from src.core.detect.yolo_world_detector import YOLOWorldDetector
            detector = YOLOWorldDetector(
                model_path=args.model or "models/yolov8s-world.pt",
                box_threshold=rule.box_threshold,
                text_threshold=rule.text_threshold,
                device=args.device,
            )
        else:
            from src.core.detect.grounding_dino_detector import GroundingDINODetector
            detector = GroundingDINODetector(
                box_threshold=rule.box_threshold,
                text_threshold=rule.text_threshold,
                device=args.device,
            )

        # ── Tracker + Rule Engine ─────────────────────────────────────────────
        tracker = ByteTracker(
            track_thresh=0.4, track_buffer=90,
            match_thresh=0.5, frame_rate=fps_src,
        )
        engine = RuleEngineV1(rules=[rule], camera_id=args.camera_id)

        _pipeline_status["state"] = "running"
        print("[PIPELINE] Started — waiting for WebSocket clients...")

        frame_id        = 0
        det_cache       = []
        last_det_frame  = -999
        emitted_ids     = set()
        t_fps           = time.time()
        fps_counter     = 0
        current_fps     = 0.0

        # JPEG encode params — lower quality = smaller payload = lower latency
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

        while not _stop_event.is_set():
            ok, frame = video_src.read()
            if not ok:
                print("[PIPELINE] Video ended.")
                _pipeline_status["state"] = "ended"
                break

            frame_id  += 1
            timestamp  = frame_id / fps_src

            # ── Detection ─────────────────────────────────────────────────────
            if frame_id % args.det_interval == 0:
                det_cache      = detector.detect(frame, [rule.prompt_positive], frame_id)
                last_det_frame = frame_id
                if det_cache:
                    print(f"[DETECT] frame={frame_id}  → {len(det_cache)} obj(s)")

            # ── Tracking ──────────────────────────────────────────────────────
            stale     = frame_id - last_det_frame
            det_input = det_cache if stale < args.det_interval else []
            tracks    = tracker.update(det_input, frame_id, timestamp)

            # ── Rule Engine ───────────────────────────────────────────────────
            engine.evaluate(tracks, frame_id, timestamp)

            # ── New Events ────────────────────────────────────────────────────
            new_evts = [e for e in engine.events if e.event_id not in emitted_ids]
            for evt in new_evts:
                emitted_ids.add(evt.event_id)
                print(f"[EVENT] {evt.event_id}  rule={evt.rule_id}  track={evt.track_id}")
                try:
                    _event_queue.put_nowait(evt.to_dict())
                except queue.Full:
                    pass  # drop if no client consuming

            # ── FPS ───────────────────────────────────────────────────────────
            fps_counter += 1
            if fps_counter >= 15:
                current_fps = fps_counter / (time.time() - t_fps)
                t_fps       = time.time()
                fps_counter = 0

            # ── Annotate frame ────────────────────────────────────────────────
            vis = _annotate_frame(frame, tracks, engine, rule,
                                  frame_id, current_fps)

            # ── Resize before sending (reduce bandwidth) ──────────────────────
            if args.stream_width > 0:
                h, w = vis.shape[:2]
                scale = args.stream_width / w
                vis = cv2.resize(vis, (args.stream_width, int(h * scale)))

            # ── Encode to JPEG ────────────────────────────────────────────────
            _, jpeg_buf = cv2.imencode(".jpg", vis, encode_params)
            jpeg_bytes  = jpeg_buf.tobytes()

            meta = {
                "frame_id":  frame_id,
                "timestamp": round(timestamp, 3),
                "fps":       round(current_fps, 1),
                "tracks":    len(tracks),
                "incidents": len(engine.get_active_incidents()),
                "events":    len(engine.events),
            }

            # Push to frame queue — drop oldest if full (clients too slow)
            if _frame_queue.full():
                try:
                    _frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                _frame_queue.put_nowait((jpeg_bytes, meta))
            except queue.Full:
                pass

            # Update shared status
            _pipeline_status.update({
                "fps":       round(current_fps, 1),
                "tracks":    len(tracks),
                "incidents": len(engine.get_active_incidents()),
                "events":    len(engine.events),
                "frame_id":  frame_id,
            })

        video_src.release()

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[PIPELINE] ERROR:\n{err}")
        _pipeline_status["state"] = "error"
        _pipeline_status["error"] = str(e)

    finally:
        _pipeline_status["state"] = "stopped"
        print("[PIPELINE] Stopped.")


def _annotate_frame(frame, tracks, engine, rule, frame_id, fps):
    vis = frame.copy()

    # Draw ROI polygon
    if rule.roi and rule.roi.enabled and rule.roi.points:
        pts = np.array(rule.roi.points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)

    # Draw tracks
    confirmed_incidents = {i.track_id for i in engine.get_confirmed_incidents()}
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        if t.track_id in confirmed_incidents:
            color, thickness = (0, 60, 255), 3   # red = incident
        elif t.state == "confirmed":
            color, thickness = (0, 200, 255), 2  # cyan = normal confirmed
        else:
            color, thickness = (100, 100, 110), 1 # gray = tentative
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"#{t.track_id} {t.class_name[:6]} {t.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # HUD overlay
    hud_lines = [
        f"Frame: {frame_id}",
        f"FPS:   {fps:.1f}",
        f"Tracks: {len(tracks)}",
        f"Events: {len(engine.events)}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(vis, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return vis


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OVD Watchdog — Inference Server (Jetson)")
    parser.add_argument("--input",        required=True,         help="Video file path on Jetson")
    parser.add_argument("--rule",         required=True,         help="Rule YAML path")
    parser.add_argument("--detector",     default="yolo_world",  choices=["yolo_world","groundingdino"])
    parser.add_argument("--model",        default=None)
    parser.add_argument("--device",       default="auto",        choices=["auto","cpu","cuda","mps"])
    parser.add_argument("--camera-id",    default="cam_01")
    parser.add_argument("--det-interval", type=int,   default=5)
    parser.add_argument("--host",         default="0.0.0.0",     help="Server bind host")
    parser.add_argument("--port",         type=int,   default=8765)
    parser.add_argument("--stream-width", type=int,   default=640,
                        help="Resize frame before sending (0=original size)")
    parser.add_argument("--jpeg-quality", type=int,   default=75,
                        help="JPEG quality 1-100 (lower = faster transfer)")
    args = parser.parse_args()

    global _pipeline_args
    _pipeline_args = args

    print("=" * 60)
    print("  OVD WATCHDOG — INFERENCE SERVER")
    print("=" * 60)
    print(f"  Video    : {args.input}")
    print(f"  Rule     : {args.rule}")
    print(f"  Device   : {args.device}")
    print(f"  Bind     : {args.host}:{args.port}")
    print(f"  Stream W : {args.stream_width}px")
    print(f"  JPEG Q   : {args.jpeg_quality}")
    print("=" * 60)
    print(f"\n  Connect from laptop:")
    print(f"    python inference_client.py --host <tailscale-ip> --port {args.port}")
    print(f"  Or check status:")
    print(f"    curl http://<tailscale-ip>:{args.port}/status\n")

    # Start pipeline in background thread
    pipeline_thread = threading.Thread(
        target=run_pipeline, args=(args,), daemon=True
    )
    pipeline_thread.start()

    # Start FastAPI/uvicorn (blocks until Ctrl+C)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

    # Cleanup
    _stop_event.set()
    pipeline_thread.join(timeout=3.0)
    print("\n[SERVER] Shutdown complete.")


if __name__ == "__main__":
    main()
