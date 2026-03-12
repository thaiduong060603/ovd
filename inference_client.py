"""
inference_client.py  —  OVD Watchdog Client
Chạy trên laptop Windows. Kết nối tới Jetson qua Tailscale, hiển thị live feed.

Install (laptop):
    pip install websockets opencv-python

Run:
    python inference_client.py --host 100.x.x.x --port 8765
    python inference_client.py --host 100.x.x.x --port 8765 --save-events

Keys (khi đang xem):
    q  — quit
    s  — save current frame as screenshot
    p  — pause / resume display
"""

import argparse
import asyncio
import base64
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import websockets


# ── Display thread ─────────────────────────────────────────────────────────────

class DisplayThread(threading.Thread):
    """
    Runs cv2.imshow in a dedicated thread.
    Receives frames from async WebSocket loop via a queue.
    """
    def __init__(self, frame_queue: queue.Queue, event_queue: queue.Queue,
                 save_events: bool, window_name: str = "OVD Watchdog — Live"):
        super().__init__(daemon=True)
        self.frame_queue  = frame_queue
        self.event_queue  = event_queue
        self.save_events  = save_events
        self.window_name  = window_name
        self.stop_flag    = threading.Event()
        self.paused       = False
        self.last_frame   = None
        self.screenshot_n = 0

        # Event log
        self.events_log   = []
        if save_events:
            Path("events_received").mkdir(exist_ok=True)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # Placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Connecting to Jetson...",
                    (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (100, 200, 255), 2)
        cv2.imshow(self.window_name, placeholder)

        while not self.stop_flag.is_set():
            # Process new frames
            try:
                frame, meta = self.frame_queue.get(timeout=0.033)
                if not self.paused:
                    self.last_frame = frame
                    annotated = self._add_client_hud(frame, meta)
                    cv2.imshow(self.window_name, annotated)
            except queue.Empty:
                if self.last_frame is not None and not self.paused:
                    pass  # keep showing last frame

            # Process events
            try:
                while True:
                    evt = self.event_queue.get_nowait()
                    self._print_event(evt)
                    if self.save_events:
                        self.events_log.append(evt)
                        self._save_event_log()
            except queue.Empty:
                pass

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop_flag.set()
                break
            elif key == ord("s"):
                self._save_screenshot()
            elif key == ord("p"):
                self.paused = not self.paused
                print(f"[CLIENT] {'Paused' if self.paused else 'Resumed'}")

        cv2.destroyAllWindows()

    def _add_client_hud(self, frame, meta: dict):
        """Add connection status and meta overlay on client side."""
        vis = frame.copy()
        # Top-right: connection indicator
        h, w = vis.shape[:2]
        cv2.circle(vis, (w - 18, 18), 8, (0, 230, 80), -1)
        cv2.putText(vis, "LIVE", (w - 60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 80), 1)
        # Bottom bar
        bar_y = h - 10
        info  = (f"  Server FPS: {meta.get('fps', '--')}"
                 f"  |  Tracks: {meta.get('tracks', 0)}"
                 f"  |  Events: {meta.get('events', 0)}"
                 f"  |  Frame: {meta.get('frame_id', 0)}")
        cv2.rectangle(vis, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.putText(vis, info, (4, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if self.paused:
            cv2.putText(vis, "PAUSED", (w//2 - 50, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
        return vis

    def _print_event(self, evt: dict):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  [{ts}] EVENT RECEIVED")
        print(f"  event_id : {evt.get('event_id')}")
        print(f"  rule_id  : {evt.get('rule_id')}")
        print(f"  camera   : {evt.get('camera_id')}")
        print(f"  track_id : {evt.get('track_id')}")
        print(f"  timestamp: {evt.get('timestamp')}s")
        print(f"  action   : {evt.get('action')}")
        print(f"{'='*60}")

    def _save_screenshot(self):
        if self.last_frame is not None:
            self.screenshot_n += 1
            path = f"screenshot_{self.screenshot_n:03d}.jpg"
            cv2.imwrite(path, self.last_frame)
            print(f"[CLIENT] Screenshot saved → {path}")

    def _save_event_log(self):
        path = Path("events_received") / "events.json"
        with open(path, "w") as f:
            json.dump(self.events_log, f, indent=2)


# ── WebSocket client ───────────────────────────────────────────────────────────

async def ws_client(host: str, port: int,
                    frame_q: queue.Queue, event_q: queue.Queue,
                    display: DisplayThread):
    uri = f"ws://{host}:{port}/ws"
    reconnect_delay = 2.0

    while not display.stop_flag.is_set():
        try:
            print(f"[CLIENT] Connecting to {uri} ...")
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message (large frames)
            ) as ws:
                print(f"[CLIENT] Connected ✓")
                reconnect_delay = 2.0  # reset on success

                async for raw_msg in ws:
                    if display.stop_flag.is_set():
                        break
                    try:
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type")
                        payload  = msg.get("payload", {})

                        if msg_type == "frame":
                            # Decode base64 JPEG → numpy
                            jpeg_bytes = base64.b64decode(payload["image"])
                            arr        = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                            frame      = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                try:
                                    frame_q.put_nowait((frame, payload.get("meta", {})))
                                except queue.Full:
                                    # Drop oldest frame
                                    try:
                                        frame_q.get_nowait()
                                        frame_q.put_nowait((frame, payload.get("meta", {})))
                                    except queue.Empty:
                                        pass

                        elif msg_type == "event":
                            try:
                                event_q.put_nowait(payload)
                            except queue.Full:
                                pass

                        elif msg_type == "status":
                            state = payload.get("state", "")
                            print(f"[CLIENT] Server state: {state}")
                            if state in ("stopped", "ended", "error"):
                                print("[CLIENT] Server pipeline ended. Waiting...")

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"[CLIENT] Message error: {e}")

        except (ConnectionRefusedError, OSError):
            print(f"[CLIENT] Cannot connect to {uri} — retrying in {reconnect_delay:.0f}s...")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[CLIENT] Connection closed: {e} — reconnecting in {reconnect_delay:.0f}s...")
        except Exception as e:
            print(f"[CLIENT] Unexpected error: {e}")

        if not display.stop_flag.is_set():
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, 15.0)  # backoff up to 15s


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OVD Watchdog Client (laptop)")
    parser.add_argument("--host",         required=True,  help="Jetson Tailscale IP")
    parser.add_argument("--port",         type=int, default=8765)
    parser.add_argument("--save-events",  action="store_true",
                        help="Save received events to events_received/events.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  OVD WATCHDOG — CLIENT")
    print("=" * 60)
    print(f"  Server   : {args.host}:{args.port}")
    print(f"  Save evts: {args.save_events}")
    print("=" * 60)
    print("\n  Keys:")
    print("    q — quit")
    print("    s — save screenshot")
    print("    p — pause / resume\n")

    frame_queue = queue.Queue(maxsize=6)
    event_queue = queue.Queue(maxsize=50)

    display = DisplayThread(frame_queue, event_queue, args.save_events)
    display.start()

    # Run async WebSocket loop in main thread
    try:
        asyncio.run(ws_client(args.host, args.port,
                              frame_queue, event_queue, display))
    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped by user.")
        display.stop_flag.set()

    display.join(timeout=2.0)
    print("[CLIENT] Bye.")


if __name__ == "__main__":
    main()
