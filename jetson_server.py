"""
jetson_server.py  —  OVD Watchdog GPU Server
Chạy trên Jetson. Hoạt động như một GPU worker thuần túy.
Client hoàn toàn kiểm soát input, config, và nhận kết quả.

Install (Jetson):
    pip install fastapi "uvicorn[standard]" websockets python-multipart pyyaml

Run:
    python jetson_server.py
    python jetson_server.py --port 8765 --device cuda

Architecture:
    POST /session/start          → tạo session, load model, bắt đầu pipeline
    POST /session/stop           → dừng pipeline
    POST /session/config/update  → hot-reload rule params (không restart)
    GET  /session/status         → trạng thái hiện tại
    GET  /session/events         → lấy events đã tích lũy
    WS   /session/stream         → 2 chiều:
                                   client→server: gửi camera frames (binary JPEG)
                                   server→client: annotated frames + events JSON
"""

import argparse
import asyncio
import base64
import json
import time
import threading
import queue
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    VIDEO_FILE   = "video_file"    # client uploads a video file
    CLIENT_CAMERA = "client_camera" # client streams frames via WebSocket
    JETSON_FILE  = "jetson_file"   # video file already on Jetson (path provided)


class SessionState(str, Enum):
    IDLE      = "idle"
    LOADING   = "loading"
    RUNNING   = "running"
    STOPPED   = "stopped"
    ERROR     = "error"


class Session:
    def __init__(self):
        self.state:        SessionState = SessionState.IDLE
        self.source_type:  Optional[SourceType] = None
        self.input_path:   Optional[str] = None     # local path on Jetson
        self.rule_data:    Dict[str, Any] = {}
        self.rule_path:    Optional[str] = None      # temp yaml file
        self.detector_name: str = "yolo_world"
        self.device:       str  = "cuda"
        self.det_interval: int  = 5
        self.stream_width: int  = 640
        self.jpeg_quality: int  = 75

        # Pipeline internals
        self.stop_event    = threading.Event()
        self.pipeline_thread: Optional[threading.Thread] = None

        # Queues for WebSocket handler
        self.frame_queue:  queue.Queue = queue.Queue(maxsize=6)
        self.event_queue:  queue.Queue = queue.Queue(maxsize=100)

        # Client camera frame input queue
        self.client_frame_queue: queue.Queue = queue.Queue(maxsize=4)

        # Stats
        self.fps:       float = 0.0
        self.tracks:    int   = 0
        self.incidents: int   = 0
        self.total_events: int = 0
        self.frame_id:  int   = 0
        self.error_msg: str   = ""

        # Accumulated events for REST polling
        self.all_events = []

    def to_dict(self) -> dict:
        return {
            "state":        self.state,
            "source_type":  self.source_type,
            "input_path":   self.input_path,
            "det_interval": self.det_interval,
            "fps":          self.fps,
            "tracks":       self.tracks,
            "incidents":    self.incidents,
            "total_events": self.total_events,
            "frame_id":     self.frame_id,
            "error":        self.error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app     = FastAPI(title="OVD Watchdog — Jetson GPU Server")
session = Session()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    source_type:   str  = "video_file"     # video_file | client_camera | jetson_file
    jetson_file_path: Optional[str] = None  # only for jetson_file
    rule_yaml:     str  = ""               # raw YAML string
    detector:      str  = "yolo_world"
    device:        str  = "cuda"
    det_interval:  int  = 5
    stream_width:  int  = 640
    jpeg_quality:  int  = 75


class ConfigUpdateRequest(BaseModel):
    dwell_seconds:   Optional[float] = None
    min_confidence:  Optional[float] = None
    min_frames:      Optional[int]   = None
    box_threshold:   Optional[float] = None
    text_threshold:  Optional[float] = None
    prompt_positive: Optional[str]   = None
    cooldown_seconds: Optional[float] = None
    roi_points:      Optional[list]  = None   # [[x,y], ...] or []
    det_interval:    Optional[int]   = None


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"server": "OVD Watchdog GPU Server", "version": "1.0"}


@app.get("/session/status")
def get_status():
    return JSONResponse(session.to_dict())


@app.get("/session/events")
def get_events(since_index: int = 0):
    """Return events from index onward (for polling clients)."""
    return JSONResponse({
        "events": session.all_events[since_index:],
        "total":  len(session.all_events),
    })


@app.post("/session/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Client uploads a video file to Jetson.
    Returns the server-side path to use in /session/start.
    """
    if session.state == SessionState.RUNNING:
        raise HTTPException(400, "Pipeline is running. Stop first.")
    # Save to temp directory
    tmp_dir = Path(tempfile.gettempdir()) / "ovd_uploads"
    tmp_dir.mkdir(exist_ok=True)
    save_path = tmp_dir / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session.input_path = str(save_path)

    file_size_mb = save_path.stat().st_size / 1024**2
    print(f"[UPLOAD] Received: {file.filename}  ({file_size_mb:.1f} MB) → {save_path}")

    return JSONResponse({
        "status":    "ok",
        "path":      str(save_path),
        "filename":  file.filename,
        "size_mb":   round(file_size_mb, 2),
    })


@app.post("/session/start")
def start_session(req: StartRequest):
    if session.state == SessionState.RUNNING:
        raise HTTPException(400, "Session already running. Stop first.")
    
    # ── Validate source ───────────────────────────────────────────────────────
    if req.source_type == SourceType.JETSON_FILE:
        if not req.jetson_file_path:
            raise HTTPException(400, "jetson_file_path required for jetson_file source.")
        if not Path(req.jetson_file_path).exists():
            raise HTTPException(404, f"File not found on Jetson: {req.jetson_file_path}")
        session.input_path = req.jetson_file_path

    elif req.source_type == SourceType.VIDEO_FILE:
        # Client must have uploaded via /session/upload_video first
        if not session.input_path:
            raise HTTPException(400, "Upload a video first via POST /session/upload_video")

    elif req.source_type == SourceType.CLIENT_CAMERA:
        session.input_path = None   # frames come via WebSocket

    # ── Parse rule YAML ───────────────────────────────────────────────────────
    if not req.rule_yaml.strip():
        raise HTTPException(400, "rule_yaml is required.")
    try:
        rule_data = yaml.safe_load(req.rule_yaml)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    # Save to temp file
    tmp_rule = Path(tempfile.gettempdir()) / "ovd_active_rule.yaml"
    with open(tmp_rule, "w") as f:
        yaml.dump(rule_data, f, default_flow_style=False, allow_unicode=True)

    # ── Store config ──────────────────────────────────────────────────────────
    session.source_type    = req.source_type
    session.rule_data      = rule_data
    session.rule_path      = str(tmp_rule)
    session.detector_name  = req.detector
    session.device         = req.device
    session.det_interval   = req.det_interval
    session.stream_width   = req.stream_width
    session.jpeg_quality   = req.jpeg_quality
    session.state          = SessionState.LOADING
    session.error_msg      = ""
    session.all_events     = []
    session.frame_id       = 0

    # Reset queues
    _clear_queue(session.frame_queue)
    _clear_queue(session.event_queue)
    _clear_queue(session.client_frame_queue)

    # ── Start pipeline thread ─────────────────────────────────────────────────
    session.stop_event.clear()
    session.pipeline_thread = threading.Thread(
        target=_pipeline_worker, daemon=True
    )
    session.pipeline_thread.start()

    print(f"[SERVER] Session started — source={req.source_type}  rule={rule_data.get('rule_id')}")
    return JSONResponse({"status": "ok", "session": session.to_dict()})


@app.post("/session/stop")
def stop_session():
    if session.state not in (SessionState.RUNNING, SessionState.LOADING):
        return JSONResponse({"status": "not_running"})
    session.stop_event.set()
    if session.pipeline_thread:
        session.pipeline_thread.join(timeout=5.0)
    session.state = SessionState.STOPPED
    print("[SERVER] Session stopped by client.")
    return JSONResponse({"status": "ok"})


@app.post("/session/config/update")
def update_config(req: ConfigUpdateRequest):
    """
    Hot-reload rule parameters while pipeline is running.
    No restart needed — changes apply on next frame evaluation.
    """
    if not session.rule_data:
        raise HTTPException(400, "No active session.")

    cond = session.rule_data.setdefault("conditions", {})
    det  = session.rule_data.setdefault("detection", {})
    act  = session.rule_data.setdefault("actions", {})
    roi  = session.rule_data.setdefault("roi", {})

    changed = []
    if req.dwell_seconds   is not None: cond["dwell_seconds"]   = req.dwell_seconds;  changed.append("dwell_seconds")
    if req.min_confidence  is not None: cond["min_confidence"]  = req.min_confidence; changed.append("min_confidence")
    if req.min_frames      is not None: cond["min_frames"]      = req.min_frames;     changed.append("min_frames")
    if req.box_threshold   is not None: det["box_threshold"]    = req.box_threshold;  changed.append("box_threshold")
    if req.text_threshold  is not None: det["text_threshold"]   = req.text_threshold; changed.append("text_threshold")
    if req.prompt_positive is not None: det["prompt_positive"]  = req.prompt_positive;changed.append("prompt_positive")
    if req.cooldown_seconds is not None: act["cooldown_seconds"] = req.cooldown_seconds; changed.append("cooldown_seconds")
    if req.det_interval    is not None: session.det_interval    = req.det_interval;   changed.append("det_interval")

    if req.roi_points is not None:
        roi["enabled"] = len(req.roi_points) > 0
        roi["type"]    = "polygon"
        roi["points"]  = req.roi_points
        changed.append("roi")

    # Rewrite temp rule file so pipeline thread picks up changes
    with open(session.rule_path, "w") as f:
        yaml.dump(session.rule_data, f, default_flow_style=False, allow_unicode=True)

    # Signal pipeline to hot-reload
    session._config_updated = True

    print(f"[SERVER] Config updated: {changed}")
    return JSONResponse({"status": "ok", "changed": changed})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — bidirectional stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/session/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    client_addr = ws.client
    print(f"[WS] Client connected: {client_addr}")

    # Send current status on connect
    await ws.send_text(json.dumps({
        "type":    "status",
        "payload": session.to_dict()
    }))

    # Receiver task: handle frames sent from client camera
    async def receive_client_frames():
        try:
            async for msg in ws.iter_bytes():
                # Binary message = JPEG frame from client camera
                arr   = np.frombuffer(msg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if session.client_frame_queue.full():
                        try:
                            session.client_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        session.client_frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
        except Exception:
            pass

    # Sender task: push annotated frames + events to client
    async def send_results():
        try:
            while True:
                # Send annotated frame
                try:
                    jpeg_bytes, meta = session.frame_queue.get_nowait()
                    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                    await ws.send_text(json.dumps({
                        "type":    "frame",
                        "payload": {"image": b64, "meta": meta}
                    }))
                except queue.Empty:
                    pass

                # Send events
                try:
                    while True:
                        evt = session.event_queue.get_nowait()
                        await ws.send_text(json.dumps({
                            "type":    "event",
                            "payload": evt
                        }))
                except queue.Empty:
                    pass

                await asyncio.sleep(0.016)   # ~60fps push cadence
        except Exception:
            pass

    try:
        await asyncio.gather(
            receive_client_frames(),
            send_results(),
        )
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {client_addr}")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker (background thread on Jetson)
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_worker():
    global session
    try:
        import sys, os
        sys.path.insert(0, os.getcwd())

        from src.core.track.byte_tracker import ByteTracker
        from src.core.rules.rule_engine_core_v1 import RuleEngineV1
        from src.models.rule import Rule

        # ── Load rule ─────────────────────────────────────────────────────────
        rule = Rule.from_yaml(session.rule_path)
        print(f"[PIPELINE] Rule: [{rule.rule_id}] {rule.description}")

        # ── Detector ──────────────────────────────────────────────────────────
        if session.detector_name == "yolo_world":
            from src.core.detect.yolo_world_detector import YOLOWorldDetector
            detector = YOLOWorldDetector(
                model_path="models/yolov8s-world.pt",
                box_threshold=rule.box_threshold,
                text_threshold=rule.text_threshold,
                device=session.device,
            )
        else:
            from src.core.detect.grounding_dino_detector import GroundingDINODetector
            detector = GroundingDINODetector(
                box_threshold=rule.box_threshold,
                text_threshold=rule.text_threshold,
                device=session.device,
            )

        # ── Video source (for file-based input) ───────────────────────────────
        video_src = None
        if session.source_type in (SourceType.VIDEO_FILE, SourceType.JETSON_FILE):
            from src.core.ingest.video_source import create_video_source
            video_src = create_video_source(session.input_path)
            fps_src   = video_src.fps or 25.0
            print(f"[PIPELINE] Video: {session.input_path}  FPS={fps_src:.1f}")
        else:
            fps_src = 30.0   # client camera assumed 30fps
            print("[PIPELINE] Waiting for client camera frames via WebSocket...")

        # ── Tracker + Engine ──────────────────────────────────────────────────
        tracker = ByteTracker(
            track_thresh=0.4, track_buffer=90,
            match_thresh=0.5, frame_rate=fps_src,
        )
        engine = RuleEngineV1(rules=[rule], camera_id="jetson_01")

        session.state   = SessionState.RUNNING
        emitted_ids     = set()
        det_cache       = []
        last_det_frame  = -999
        t_fps           = time.time()
        fps_counter     = 0
        current_fps     = 0.0
        encode_params   = [cv2.IMWRITE_JPEG_QUALITY, session.jpeg_quality]
        session._config_updated = False

        print("[PIPELINE] Running.")

        while not session.stop_event.is_set():

            # ── Hot-reload config if updated ──────────────────────────────────
            if getattr(session, "_config_updated", False):
                try:
                    new_rule = Rule.from_yaml(session.rule_path)
                    engine.rules = [new_rule]
                    rule = new_rule
                    # Update detector thresholds
                    detector.box_threshold  = rule.box_threshold
                    detector.text_threshold = rule.text_threshold
                    session._config_updated = False
                    print("[PIPELINE] Config hot-reloaded ✓")
                except Exception as e:
                    print(f"[PIPELINE] Hot-reload error: {e}")

            # ── Get next frame ─────────────────────────────────────────────────
            if session.source_type == SourceType.CLIENT_CAMERA:
                try:
                    frame = session.client_frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue   # wait for client to send frames
            else:
                ok, frame = video_src.read()
                if not ok:
                    print("[PIPELINE] Video ended.")
                    session.state = SessionState.STOPPED
                    break

            session.frame_id += 1
            timestamp = session.frame_id / fps_src

            # ── Detection ─────────────────────────────────────────────────────
            if session.frame_id % session.det_interval == 0:
                det_cache      = detector.detect(frame, [rule.prompt_positive], session.frame_id)
                last_det_frame = session.frame_id

            # ── Tracking ──────────────────────────────────────────────────────
            stale     = session.frame_id - last_det_frame
            det_input = det_cache if stale < session.det_interval else []
            tracks    = tracker.update(det_input, session.frame_id, timestamp)

            # ── Rule evaluation ───────────────────────────────────────────────
            engine.evaluate(tracks, session.frame_id, timestamp)

            # ── Emit new events ───────────────────────────────────────────────
            new_evts = [e for e in engine.events if e.event_id not in emitted_ids]
            for evt in new_evts:
                emitted_ids.add(evt.event_id)
                evt_dict = evt.to_dict()
                session.all_events.append(evt_dict)
                try:
                    session.event_queue.put_nowait(evt_dict)
                except queue.Full:
                    pass
                print(f"[EVENT] {evt.event_id}  rule={evt.rule_id}  track={evt.track_id}")

            # ── FPS ───────────────────────────────────────────────────────────
            fps_counter += 1
            if fps_counter >= 15:
                current_fps   = fps_counter / (time.time() - t_fps)
                t_fps         = time.time()
                fps_counter   = 0
                session.fps   = round(current_fps, 1)

            session.tracks    = len(tracks)
            session.incidents = len(engine.get_active_incidents())
            session.total_events = len(engine.events)

            # ── Annotate + encode frame ────────────────────────────────────────
            vis = _annotate(frame, tracks, engine, rule,
                            session.frame_id, current_fps)

            if session.stream_width > 0:
                h, w  = vis.shape[:2]
                scale = session.stream_width / w
                vis   = cv2.resize(vis, (session.stream_width, int(h * scale)))

            _, jpeg_buf = cv2.imencode(".jpg", vis, encode_params)

            meta = {
                "frame_id":  session.frame_id,
                "timestamp": round(timestamp, 3),
                "fps":       session.fps,
                "tracks":    session.tracks,
                "incidents": session.incidents,
                "events":    session.total_events,
            }

            # Drop oldest frame if queue full (client too slow)
            if session.frame_queue.full():
                try:
                    session.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                session.frame_queue.put_nowait((jpeg_buf.tobytes(), meta))
            except queue.Full:
                pass

        if video_src:
            video_src.release()

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[PIPELINE] ERROR:\n{err}")
        session.state     = SessionState.ERROR
        session.error_msg = str(e)
    finally:
        if session.state == SessionState.RUNNING:
            session.state = SessionState.STOPPED
        print("[PIPELINE] Worker exited.")


def _annotate(frame, tracks, engine, rule, frame_id, fps):
    vis = frame.copy()
    # ROI overlay
    if rule.roi and rule.roi.enabled and rule.roi.points:
        pts = np.array(rule.roi.points, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)
        cv2.polylines(vis, [pts], True, (0, 200, 255), 2)

    # Tracks
    incident_track_ids = {i.track_id for i in engine.get_confirmed_incidents()}
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        if t.track_id in incident_track_ids:
            color, thick = (0, 60, 255), 3
        elif t.state == "confirmed":
            color, thick = (0, 200, 255), 2
        else:
            color, thick = (100, 100, 110), 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, f"#{t.track_id} {t.class_name[:6]} {t.confidence:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # HUD
    for i, txt in enumerate([
        f"Frame: {frame_id}",
        f"FPS:   {fps:.1f}",
        f"Tracks: {len(tracks)}",
        f"Events: {len(engine.events)}",
    ]):
        cv2.putText(vis, txt, (8, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return vis


def _clear_queue(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OVD Watchdog — Jetson GPU Server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8765)
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu","auto"])
    args = parser.parse_args()

    print("=" * 60)
    print("  OVD WATCHDOG — JETSON GPU SERVER")
    print("=" * 60)
    print(f"  Bind   : {args.host}:{args.port}")
    print(f"  Device : {args.device}")
    print()
    print("  Endpoints:")
    print(f"    POST  http://<ip>:{args.port}/session/upload_video")
    print(f"    POST  http://<ip>:{args.port}/session/start")
    print(f"    POST  http://<ip>:{args.port}/session/stop")
    print(f"    POST  http://<ip>:{args.port}/session/config/update")
    print(f"    GET   http://<ip>:{args.port}/session/status")
    print(f"    GET   http://<ip>:{args.port}/session/events")
    print(f"    WS    ws://<ip>:{args.port}/session/stream")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
