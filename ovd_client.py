"""
ovd_client.py  —  OVD Watchdog Client
Chạy trên PC / Laptop. Kết nối tới Jetson GPU Server qua Tailscale.

Install (laptop Windows):
    pip install websockets opencv-python requests pyyaml

Usage:
    # Gửi video file lên Jetson rồi chạy
    python ovd_client.py --host 100.x.x.x --video cross1.mp4 --rule my_rule.yaml

    # Dùng camera của laptop (stream lên Jetson xử lý)
    python ovd_client.py --host 100.x.x.x --camera 0 --rule my_rule.yaml

    # Dùng video file đã có sẵn trên Jetson
    python ovd_client.py --host 100.x.x.x --jetson-file /app/videos/cross1.mp4 --rule my_rule.yaml

    # Cập nhật config đang chạy (không cần restart)
    python ovd_client.py --host 100.x.x.x --update-config --dwell 3.0 --confidence 0.35

Keys (khi xem live):
    q  — quit + stop server pipeline
    s  — screenshot
    p  — pause/resume display
    +  — tăng dwell_seconds +0.5
    -  — giảm dwell_seconds -0.5
"""

import argparse
import asyncio
import base64
import json
import sys
import time
import threading
import queue
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import requests
import websockets


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def api(base_url: str, method: str, path: str, **kwargs):
    url = f"{base_url}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"[CLIENT] Cannot connect to {url}")
        print(f"         Is Jetson server running? Check Tailscale IP.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[CLIENT] HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)


def upload_video(base_url: str, video_path: str) -> str:
    path = Path(video_path)
    size_mb = path.stat().st_size / 1024**2
    print(f"[CLIENT] Uploading {path.name} ({size_mb:.1f} MB) to Jetson...")

    with open(path, "rb") as f:
        resp = requests.post(
            f"{base_url}/session/upload_video",
            files={"file": (path.name, f, "video/mp4")},
            timeout=120,   # large files need more time
        )
    resp.raise_for_status()
    result = resp.json()
    print(f"[CLIENT] Upload complete → server path: {result['path']}")
    return result["path"]


def start_session(base_url: str, source_type: str,
                  rule_yaml_str: str, jetson_file_path: str = None,
                  detector: str = "yolo_world", device: str = "cuda",
                  det_interval: int = 5, stream_width: int = 640,
                  jpeg_quality: int = 75):
    payload = {
        "source_type":       source_type,
        "rule_yaml":         rule_yaml_str,
        "detector":          detector,
        "device":            device,
        "det_interval":      det_interval,
        "stream_width":      stream_width,
        "jpeg_quality":      jpeg_quality,
    }
    if jetson_file_path:
        payload["jetson_file_path"] = jetson_file_path

    print(f"[CLIENT] Starting session (source={source_type})...")
    result = api(base_url, "post", "/session/start", json=payload)
    print(f"[CLIENT] Session started ✓  state={result.get('session', {}).get('state')}")
    return result


def stop_session(base_url: str):
    result = api(base_url, "post", "/session/stop")
    print(f"[CLIENT] Session stopped: {result.get('status')}")


def update_config(base_url: str, **kwargs):
    # Remove None values
    payload = {k: v for k, v in kwargs.items() if v is not None}
    result  = api(base_url, "post", "/session/config/update", json=payload)
    print(f"[CLIENT] Config updated: {result.get('changed')}")
    return result


def check_status(base_url: str):
    result = api(base_url, "get", "/session/status")
    print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Display thread
# ─────────────────────────────────────────────────────────────────────────────

class LiveDisplay(threading.Thread):
    def __init__(self, frame_q: queue.Queue, event_q: queue.Queue,
                 base_url: str, rule_data: dict):
        super().__init__(daemon=True)
        self.frame_q   = frame_q
        self.event_q   = event_q
        self.base_url  = base_url
        self.rule_data = rule_data
        self.stop_flag = threading.Event()
        self.paused    = False
        self.last_frame = None
        self.shot_n     = 0
        # Current adjustable params (shown in HUD)
        self.dwell_seconds = rule_data.get("conditions", {}).get("dwell_seconds", 2.0)

    def run(self):
        cv2.namedWindow("OVD Watchdog", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OVD Watchdog", 900, 620)

        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Connecting to Jetson...",
                    (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.imshow("OVD Watchdog", placeholder)

        while not self.stop_flag.is_set():
            # New frame
            try:
                frame, meta = self.frame_q.get(timeout=0.04)
                if not self.paused:
                    self.last_frame = frame
                    vis = self._overlay(frame, meta)
                    cv2.imshow("OVD Watchdog", vis)
            except queue.Empty:
                pass

            # Events
            try:
                while True:
                    evt = self.event_q.get_nowait()
                    self._print_event(evt)
            except queue.Empty:
                pass

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[CLIENT] Quit — stopping server pipeline...")
                stop_session(self.base_url)
                self.stop_flag.set()
                break
            elif key == ord("s"):
                self._screenshot()
            elif key == ord("p"):
                self.paused = not self.paused
                print(f"[CLIENT] {'Paused' if self.paused else 'Resumed'}")
            elif key == ord("+") or key == ord("="):
                self.dwell_seconds = round(self.dwell_seconds + 0.5, 1)
                update_config(self.base_url, dwell_seconds=self.dwell_seconds)
            elif key == ord("-"):
                self.dwell_seconds = max(0.1, round(self.dwell_seconds - 0.5, 1))
                update_config(self.base_url, dwell_seconds=self.dwell_seconds)

        cv2.destroyAllWindows()

    # Add draw annotations when receive frames from jetson
    def _draw_annotations(self, frame, meta):
        vis = frame.copy()
        # Vẽ tracks
        for t in meta.get("tracks_data", []):
            x1,y1,x2,y2 = t["bbox"]
            color = (0,60,255) if t["is_incident"] else (0,200,255)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 3)
            cv2.putText(vis, f"#{t['track_id']} {t['class_name'][:8]}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vẽ ROI
        if meta.get("roi_points"):
            pts = np.array(meta["roi_points"], np.int32)
            cv2.polylines(vis, [pts], True, (0,200,255), 2)
        
        return vis
    def _overlay(self, frame, meta):
        # vis = frame.copy()
        vis = self._draw_annotations(frame, meta)
        h, w = vis.shape[:2]
        # Green dot = connected
        cv2.circle(vis, (w - 18, 18), 8, (0, 230, 80), -1)
        cv2.putText(vis, "LIVE", (w - 58, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 80), 1)
        # Bottom info bar
        cv2.rectangle(vis, (0, h - 30), (w, h), (15, 15, 15), -1)
        bar = (f"  Server FPS: {meta.get('fps','--')}"
               f"  Tracks: {meta.get('tracks',0)}"
               f"  Events: {meta.get('events',0)}"
               f"  Frame: {meta.get('frame_id',0)}"
               f"  dwell={self.dwell_seconds}s  [+/-] adjust  [s] shot  [q] quit")
        cv2.putText(vis, bar, (4, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        if self.paused:
            cv2.putText(vis, "PAUSED", (w//2 - 60, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 255), 3)
        return vis

    def _print_event(self, evt: dict):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*55}")
        print(f"  [{ts}]  ⚠  ALERT RECEIVED")
        print(f"  rule     : {evt.get('rule_id')}")
        print(f"  track_id : {evt.get('track_id')}")
        print(f"  camera   : {evt.get('camera_id')}")
        print(f"  action   : {evt.get('action')}")
        print(f"  at       : {evt.get('timestamp')}s  frame={evt.get('evidence',{}).get('frame_id')}")
        print(f"{'='*55}")

    def _screenshot(self):
        if self.last_frame is not None:
            self.shot_n += 1
            fn = f"ovd_screenshot_{self.shot_n:03d}.jpg"
            cv2.imwrite(fn, self.last_frame)
            print(f"[CLIENT] Screenshot → {fn}")


# ─────────────────────────────────────────────────────────────────────────────
# Camera streamer (client camera → Jetson)
# ─────────────────────────────────────────────────────────────────────────────

async def stream_camera(ws_url: str, cam_index: int,
                        frame_q: queue.Queue, event_q: queue.Queue,
                        stop_flag: threading.Event, jpeg_quality: int = 70):
    """
    Open laptop camera, encode each frame as JPEG,
    send to Jetson over WebSocket (binary),
    receive annotated frames + events back.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[CLIENT] Cannot open camera {cam_index}")
        return

    print(f"[CLIENT] Camera {cam_index} opened ✓")
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    try:
        async with websockets.connect(
            ws_url, max_size=10 * 1024 * 1024, ping_interval=20
        ) as ws:
            print(f"[CLIENT] WebSocket connected ✓")

            async def send_frames():
                while not stop_flag.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    _, buf = cv2.imencode(".jpg", frame, encode_params)
                    await ws.send(buf.tobytes())
                    await asyncio.sleep(0.033)  # ~30fps

            async def recv_results():
                async for raw in ws:
                    if stop_flag.is_set():
                        break
                    if isinstance(raw, str):
                        _handle_message(raw, frame_q, event_q)

            await asyncio.gather(send_frames(), recv_results())

    finally:
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Video file / server-side stream receiver
# ─────────────────────────────────────────────────────────────────────────────

async def receive_stream(ws_url: str, frame_q: queue.Queue,
                         event_q: queue.Queue, stop_flag: threading.Event):
    """
    For video_file / jetson_file sources:
    Just connect and receive annotated frames + events.
    """
    reconnect_delay = 2.0
    while not stop_flag.is_set():
        try:
            print(f"[CLIENT] Connecting to {ws_url} ...")
            async with websockets.connect(
                ws_url, max_size=10 * 1024 * 1024, ping_interval=20
            ) as ws:
                print("[CLIENT] WebSocket connected ✓")
                reconnect_delay = 2.0
                async for raw in ws:
                    if stop_flag.is_set():
                        break
                    if isinstance(raw, str):
                        done = _handle_message(raw, frame_q, event_q)
                        if done:
                            stop_flag.set()
                            break
        except (ConnectionRefusedError, OSError):
            print(f"[CLIENT] Cannot connect — retrying in {reconnect_delay:.0f}s...")
        except websockets.exceptions.ConnectionClosed as e:
            if not stop_flag.is_set():
                print(f"[CLIENT] Connection closed — retrying in {reconnect_delay:.0f}s...")
        except Exception as e:
            print(f"[CLIENT] WS error: {e}")
        if not stop_flag.is_set():
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, 15.0)


def _handle_message(raw, frame_q: queue.Queue, event_q: queue.Queue):
    """Xử lý cả text và binary"""
    if isinstance(raw, bytes):
        if raw.startswith(b"FRAME"):
            idx = raw.find(b"__META__")
            jpeg_bytes = raw[5:idx]
            meta_bytes = raw[idx + 8:]
            meta = json.loads(meta_bytes.decode("utf-8"))

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if frame_q.full():
                    frame_q.get_nowait()
                frame_q.put_nowait((frame, meta))
        # Sau này sẽ thêm EVIDENCE ở đây (xem phần lưu alert)

    elif isinstance(raw, str):
        try:
            msg = json.loads(raw)
            if msg.get("type") == "event":
                event_q.put_nowait(msg.get("payload", {}))
            elif msg.get("type") == "status":
                print(f"[CLIENT] Server state: {msg['payload'].get('state')}")
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OVD Watchdog Client")
    parser.add_argument("--host",     required=True, help="Jetson Tailscale IP")
    parser.add_argument("--port",     type=int, default=8765)

    # Input sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video",       help="Local video file to upload and run on Jetson")
    src.add_argument("--camera",      type=int, metavar="INDEX",
                     help="Laptop camera index (stream to Jetson)")
    src.add_argument("--jetson-file", help="Video file already on Jetson (full path)")

    # Rule
    parser.add_argument("--rule",     help="Rule YAML file path (local)")

    # Pipeline options
    parser.add_argument("--detector",     default="yolo_world",
                        choices=["yolo_world","groundingdino"])
    parser.add_argument("--device",       default="cuda",
                        choices=["cuda","cpu","auto"])
    parser.add_argument("--det-interval", type=int,   default=5)
    parser.add_argument("--stream-width", type=int,   default=480)
    parser.add_argument("--jpeg-quality", type=int,   default=30)

    # Utility commands
    parser.add_argument("--status",        action="store_true", help="Print server status and exit")
    parser.add_argument("--stop",          action="store_true", help="Stop server pipeline and exit")
    parser.add_argument("--update-config", action="store_true", help="Update config only (no stream)")
    parser.add_argument("--dwell",         type=float, help="New dwell_seconds")
    parser.add_argument("--confidence",    type=float, help="New min_confidence")
    parser.add_argument("--box-threshold", type=float, help="New box_threshold")
    parser.add_argument("--det-interval-update", type=int, help="New det_interval")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    ws_url   = f"ws://{args.host}:{args.port}/session/stream"

    # ── Utility commands ──────────────────────────────────────────────────────
    if args.status:
        check_status(base_url)
        return

    if args.stop:
        stop_session(base_url)
        return

    if args.update_config:
        update_config(base_url,
                      dwell_seconds=args.dwell,
                      min_confidence=args.confidence,
                      box_threshold=args.box_threshold,
                      det_interval=args.det_interval_update)
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    if not args.rule:
        print("[CLIENT] --rule is required.")
        sys.exit(1)

    rule_text = Path(args.rule).read_text(encoding="utf-8")
    rule_data = {}
    try:
        import yaml
        rule_data = yaml.safe_load(rule_text)
    except Exception:
        pass

    # Determine source type and prepare
    if args.video:
        jetson_path = upload_video(base_url, args.video)
        source_type = "video_file"
    elif args.jetson_file:
        jetson_path = args.jetson_file
        source_type = "jetson_file"
    elif args.camera is not None:
        jetson_path = None
        source_type = "client_camera"
    else:
        print("[CLIENT] Specify one of: --video, --camera, --jetson-file")
        sys.exit(1)

    # Start session on Jetson
    start_session(
        base_url        = base_url,
        source_type     = source_type,
        rule_yaml_str   = rule_text,
        jetson_file_path= jetson_path,
        detector        = args.detector,
        device          = args.device,
        det_interval    = args.det_interval,
        stream_width    = args.stream_width,
        jpeg_quality    = args.jpeg_quality,
    )

    # Setup display
    frame_q = queue.Queue(maxsize=6)
    event_q = queue.Queue(maxsize=50)
    display = LiveDisplay(frame_q, event_q, base_url, rule_data)
    display.start()

    print("\n  Keys: [q] quit  [s] screenshot  [p] pause  [+/-] dwell\n")

    # Run WebSocket loop
    try:
        if source_type == "client_camera":
            asyncio.run(stream_camera(ws_url, args.camera,
                                      frame_q, event_q,
                                      display.stop_flag,
                                      args.jpeg_quality))
        else:
            asyncio.run(receive_stream(ws_url, frame_q, event_q,
                                       display.stop_flag))
    except KeyboardInterrupt:
        print("\n[CLIENT] Interrupted — stopping server...")
        stop_session(base_url)
        display.stop_flag.set()

    display.join(timeout=2.0)
    print("[CLIENT] Done.")


if __name__ == "__main__":
    main()
