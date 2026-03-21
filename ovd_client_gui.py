"""
ovd_client_gui.py  —  OVD Watchdog Client (Tkinter GUI)
Wraps all functionality from ovd_client.py into a dark, industrial-style GUI.

Install:
    pip install websockets opencv-python requests pyyaml pillow

Usage:
    python ovd_client_gui.py
"""

import asyncio
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2
import numpy as np

# ── try optional deps gracefully ──────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Re-used backend logic (ported from ovd_client.py — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def api(base_url: str, method: str, path: str, **kwargs):
    url = f"{base_url}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")


def upload_video(base_url: str, video_path: str, progress_cb=None) -> str:
    path = Path(video_path)
    size_mb = path.stat().st_size / 1024**2
    if progress_cb:
        progress_cb(f"Uploading {path.name} ({size_mb:.1f} MB)...")
    with open(path, "rb") as f:
        resp = requests.post(
            f"{base_url}/session/upload_video",
            files={"file": (path.name, f, "video/mp4")},
            timeout=120,
        )
    resp.raise_for_status()
    result = resp.json()
    if progress_cb:
        progress_cb(f"Upload complete → {result['path']}")
    return result["path"]


def start_session(base_url, source_type, rule_yaml_str, jetson_file_path=None,
                  detector="yolo_world", device="cuda",
                  det_interval=5, stream_width=640, jpeg_quality=75):
    payload = {
        "source_type":   source_type,
        "rule_yaml":     rule_yaml_str,
        "detector":      detector,
        "device":        device,
        "det_interval":  det_interval,
        "stream_width":  stream_width,
        "jpeg_quality":  jpeg_quality,
    }
    if jetson_file_path:
        payload["jetson_file_path"] = jetson_file_path
    return api(base_url, "post", "/session/start", json=payload)


def stop_session(base_url: str):
    return api(base_url, "post", "/session/stop")


def update_config(base_url: str, **kwargs):
    payload = {k: v for k, v in kwargs.items() if v is not None}
    return api(base_url, "post", "/session/config/update", json=payload)


def check_status(base_url: str):
    return api(base_url, "get", "/session/status")


def _handle_message(raw, frame_q: queue.Queue, event_q: queue.Queue):
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
    elif isinstance(raw, str):
        try:
            msg = json.loads(raw)
            if msg.get("type") == "event":
                event_q.put_nowait(msg.get("payload", {}))
            elif msg.get("type") == "status":
                pass  # handled via log
        except Exception:
            pass


async def receive_stream(ws_url, frame_q, event_q, stop_flag):
    reconnect_delay = 2.0
    while not stop_flag.is_set():
        try:
            async with websockets.connect(ws_url, max_size=10*1024*1024, ping_interval=20) as ws:
                reconnect_delay = 2.0
                async for raw in ws:
                    if stop_flag.is_set():
                        break
                    done = _handle_message(raw, frame_q, event_q)
                    if done:
                        stop_flag.set()
                        break
        except (ConnectionRefusedError, OSError):
            pass
        except Exception:
            pass
        if not stop_flag.is_set():
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, 15.0)


async def stream_camera(ws_url, cam_index, frame_q, event_q, stop_flag, jpeg_quality=70):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    try:
        async with websockets.connect(ws_url, max_size=10*1024*1024, ping_interval=20) as ws:
            async def send_frames():
                while not stop_flag.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    _, buf = cv2.imencode(".jpg", frame, encode_params)
                    await ws.send(buf.tobytes())
                    await asyncio.sleep(0.033)

            async def recv_results():
                async for raw in ws:
                    if stop_flag.is_set():
                        break
                    if isinstance(raw, (str, bytes)):
                        _handle_message(raw, frame_q, event_q)

            await asyncio.gather(send_frames(), recv_results())
    finally:
        cap.release()


def _draw_annotations(frame, meta):
    vis = frame.copy()
    for t in meta.get("tracks_data", []):
        x1, y1, x2, y2 = t["bbox"]
        color = (0, 60, 255) if t["is_incident"] else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, f"#{t['track_id']} {t['class_name'][:8]}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if meta.get("roi_points"):
        pts = np.array(meta["roi_points"], np.int32)
        cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────

# ── colour palette ─────────────────────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#13161d"
BORDER   = "#1f2330"
ACCENT   = "#00e5ff"
ACCENT2  = "#ff5252"
TEXT     = "#d0d8e8"
MUTED    = "#5a6278"
SUCCESS  = "#00e676"
WARNING  = "#ffab40"
FONT_MONO = ("Consolas", 9)
FONT_LABEL = ("Segoe UI", 9)
FONT_HEAD  = ("Segoe UI Semibold", 10)


class OVDClientGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OVD Watchdog — Client")
        self.configure(bg=BG)
        self.geometry("1280x820")
        self.minsize(960, 640)

        # state
        self.session_running = False
        self.paused          = False
        self.stop_flag       = threading.Event()
        self.frame_q         = queue.Queue(maxsize=6)
        self.event_q         = queue.Queue(maxsize=100)
        self.dwell_seconds   = 2.0
        self.shot_n          = 0
        self.last_frame      = None
        self._ws_thread      = None

        self._build_ui()
        self._poll_frames()
        self._poll_events()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        self._style()

        # ── top bar ──────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, height=48)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(top, text="⬡  OVD WATCHDOG", fg=ACCENT, bg=PANEL,
                 font=("Consolas", 14, "bold")).pack(side="left", padx=18, pady=10)

        self.status_dot = tk.Label(top, text="●", fg=MUTED, bg=PANEL, font=("Consolas", 14))
        self.status_dot.pack(side="right", padx=6)
        self.status_label = tk.Label(top, text="IDLE", fg=MUTED, bg=PANEL, font=FONT_MONO)
        self.status_label.pack(side="right", padx=2, pady=10)

        # ── main pane (left config | right video+log) ─────────────────────
        pane = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashwidth=4, sashrelief="flat", bd=0)
        pane.pack(fill="both", expand=True, padx=0, pady=0)

        left  = self._build_left(pane)
        right = self._build_right(pane)

        pane.add(left,  minsize=310, width=340)
        pane.add(right, minsize=500)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox",
                     fieldbackground=PANEL, background=PANEL,
                     foreground=TEXT, selectbackground=BORDER,
                     bordercolor=BORDER, arrowcolor=ACCENT)
        s.configure("TScale",
                     background=BG, troughcolor=BORDER,
                     sliderrelief="flat")
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)])

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BG, pady=4, padx=6)
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        tk.Label(header, text=title.upper(), fg=ACCENT, bg=BG,
                 font=("Consolas", 8, "bold")).pack(side="left")
        sep = tk.Frame(outer, bg=BORDER, height=1)
        sep.pack(fill="x", pady=(2, 6))
        body = tk.Frame(outer, bg=BG)
        body.pack(fill="x")
        return outer, body

    def _lbl(self, parent, text):
        tk.Label(parent, text=text, fg=MUTED, bg=BG,
                 font=FONT_LABEL, anchor="w").grid(sticky="w")

    def _entry(self, parent, var, row, label, col=0, width=22, **kw):
        tk.Label(parent, text=label, fg=MUTED, bg=BG,
                 font=FONT_LABEL, anchor="w").grid(row=row, column=col,
                 sticky="w", padx=(0, 6), pady=2)
        e = tk.Entry(parent, textvariable=var, bg=PANEL, fg=TEXT,
                     insertbackground=ACCENT, relief="flat",
                     bd=0, highlightthickness=1,
                     highlightbackground=BORDER,
                     highlightcolor=ACCENT, width=width, **kw)
        e.grid(row=row, column=col+1, sticky="ew", pady=2)
        return e

    # ── left panel ────────────────────────────────────────────────────────

    def _build_left(self, pane):
        scroll_frame = tk.Frame(pane, bg=BG)

        canvas = tk.Canvas(scroll_frame, bg=BG, bd=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.itemconfig(win_id, width=canvas.winfo_width())
        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        self._build_connection(inner)
        self._build_source(inner)
        self._build_pipeline(inner)
        self._build_livectrl(inner)
        self._build_utility(inner)

        return scroll_frame

    def _build_connection(self, parent):
        card, body = self._card(parent, "Connection")
        card.pack(fill="x", padx=8, pady=(8, 0))
        body.columnconfigure(1, weight=1)

        self.var_host = tk.StringVar(value="100.x.x.x")
        self.var_port = tk.StringVar(value="8765")
        self._entry(body, self.var_host, 0, "Jetson Host", width=20)
        self._entry(body, self.var_port, 1, "Port", width=8)

    def _build_source(self, parent):
        card, body = self._card(parent, "Source")
        card.pack(fill="x", padx=8, pady=(10, 0))
        body.columnconfigure(1, weight=1)

        self.var_src = tk.StringVar(value="video_file")
        sources = [("Local Video → Upload", "video_file"),
                   ("Jetson File (on server)", "jetson_file"),
                   ("Laptop Camera", "client_camera")]
        for i, (lbl, val) in enumerate(sources):
            tk.Radiobutton(body, text=lbl, variable=self.var_src, value=val,
                           bg=BG, fg=TEXT, selectcolor=PANEL,
                           activebackground=BG, activeforeground=ACCENT,
                           font=FONT_LABEL, command=self._source_changed
                           ).grid(row=i, column=0, columnspan=2, sticky="w", pady=1)

        # video path row
        self.video_row = tk.Frame(body, bg=BG)
        self.video_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.video_row.columnconfigure(1, weight=1)
        self.var_video = tk.StringVar()
        tk.Label(self.video_row, text="File", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, padx=(0, 6))
        tk.Entry(self.video_row, textvariable=self.var_video, bg=PANEL, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, width=18
                 ).grid(row=0, column=1, sticky="ew")
        tk.Button(self.video_row, text="Browse", bg=BORDER, fg=TEXT,
                  activebackground=ACCENT, activeforeground=BG,
                  relief="flat", cursor="hand2", padx=6,
                  font=FONT_LABEL, command=self._browse_video
                  ).grid(row=0, column=2, padx=(4, 0))

        # jetson file row
        self.jfile_row = tk.Frame(body, bg=BG)
        self.jfile_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.jfile_row.columnconfigure(1, weight=1)
        self.var_jfile = tk.StringVar(value="/app/videos/")
        tk.Label(self.jfile_row, text="Path", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, padx=(0, 6))
        tk.Entry(self.jfile_row, textvariable=self.var_jfile, bg=PANEL, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT
                 ).grid(row=0, column=1, sticky="ew")

        # camera row
        self.cam_row = tk.Frame(body, bg=BG)
        self.cam_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.var_cam = tk.StringVar(value="0")
        tk.Label(self.cam_row, text="Camera Index", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, padx=(0, 6))
        tk.Entry(self.cam_row, textvariable=self.var_cam, bg=PANEL, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, width=5
                 ).grid(row=0, column=1, sticky="w")

        # rule file
        rule_row = tk.Frame(body, bg=BG)
        rule_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        rule_row.columnconfigure(1, weight=1)
        self.var_rule = tk.StringVar()
        tk.Label(rule_row, text="Rule YAML", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, padx=(0, 6))
        tk.Entry(rule_row, textvariable=self.var_rule, bg=PANEL, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, width=14
                 ).grid(row=0, column=1, sticky="ew")
        tk.Button(rule_row, text="Browse", bg=BORDER, fg=TEXT,
                  activebackground=ACCENT, activeforeground=BG,
                  relief="flat", cursor="hand2", padx=6,
                  font=FONT_LABEL, command=self._browse_rule
                  ).grid(row=0, column=2, padx=(4, 0))

        self._source_changed()

    def _build_pipeline(self, parent):
        card, body = self._card(parent, "Pipeline")
        card.pack(fill="x", padx=8, pady=(10, 0))
        body.columnconfigure(1, weight=1)

        self.var_detector = tk.StringVar(value="yolo_world")
        self.var_device   = tk.StringVar(value="cuda")
        self.var_det_int  = tk.StringVar(value="5")
        self.var_sw       = tk.StringVar(value="480")
        self.var_jq       = tk.StringVar(value="30")

        tk.Label(body, text="Detector", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(body, textvariable=self.var_detector,
                     values=["yolo_world", "groundingdino"],
                     state="readonly", width=18
                     ).grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(body, text="Device", fg=MUTED, bg=BG,
                 font=FONT_LABEL).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(body, textvariable=self.var_device,
                     values=["cuda", "cpu", "auto"],
                     state="readonly", width=10
                     ).grid(row=1, column=1, sticky="w", pady=2)

        self._entry(body, self.var_det_int, 2, "Det Interval", width=6)
        self._entry(body, self.var_sw,      3, "Stream Width",  width=6)
        self._entry(body, self.var_jq,      4, "JPEG Quality",  width=6)

    def _build_livectrl(self, parent):
        card, body = self._card(parent, "Live Controls")
        card.pack(fill="x", padx=8, pady=(10, 0))

        # dwell slider
        dwell_row = tk.Frame(body, bg=BG)
        dwell_row.pack(fill="x")
        tk.Label(dwell_row, text="Dwell (s)", fg=MUTED, bg=BG,
                 font=FONT_LABEL, width=10, anchor="w").pack(side="left")
        self.var_dwell = tk.DoubleVar(value=2.0)
        self.dwell_label = tk.Label(dwell_row, text="2.0s", fg=ACCENT, bg=BG,
                                     font=FONT_MONO, width=5)
        self.dwell_label.pack(side="right")
        sl = ttk.Scale(dwell_row, from_=0.1, to=10.0, orient="horizontal",
                       variable=self.var_dwell, command=self._dwell_changed)
        sl.pack(fill="x", expand=True, padx=(0, 6))

        # confidence
        conf_row = tk.Frame(body, bg=BG)
        conf_row.pack(fill="x", pady=(4, 0))
        tk.Label(conf_row, text="Confidence", fg=MUTED, bg=BG,
                 font=FONT_LABEL, width=10, anchor="w").pack(side="left")
        self.var_conf = tk.DoubleVar(value=0.35)
        self.conf_label = tk.Label(conf_row, text="0.35", fg=ACCENT, bg=BG,
                                    font=FONT_MONO, width=5)
        self.conf_label.pack(side="right")
        ttk.Scale(conf_row, from_=0.05, to=1.0, orient="horizontal",
                  variable=self.var_conf,
                  command=lambda v: self.conf_label.config(
                      text=f"{float(v):.2f}")
                  ).pack(fill="x", expand=True, padx=(0, 6))

        # box threshold
        bth_row = tk.Frame(body, bg=BG)
        bth_row.pack(fill="x", pady=(4, 0))
        tk.Label(bth_row, text="Box Thresh.", fg=MUTED, bg=BG,
                 font=FONT_LABEL, width=10, anchor="w").pack(side="left")
        self.var_bth = tk.DoubleVar(value=0.3)
        self.bth_label = tk.Label(bth_row, text="0.30", fg=ACCENT, bg=BG,
                                   font=FONT_MONO, width=5)
        self.bth_label.pack(side="right")
        ttk.Scale(bth_row, from_=0.05, to=1.0, orient="horizontal",
                  variable=self.var_bth,
                  command=lambda v: self.bth_label.config(
                      text=f"{float(v):.2f}")
                  ).pack(fill="x", expand=True, padx=(0, 6))

        # push update button
        tk.Button(body, text="⟳  Push Config Update",
                  bg=BORDER, fg=ACCENT,
                  activebackground=ACCENT, activeforeground=BG,
                  relief="flat", cursor="hand2", pady=5,
                  font=("Consolas", 9, "bold"),
                  command=self._push_config
                  ).pack(fill="x", pady=(8, 2))

    def _build_utility(self, parent):
        card, body = self._card(parent, "Session")
        card.pack(fill="x", padx=8, pady=(10, 0))

        btn_cfg = dict(relief="flat", cursor="hand2", pady=6,
                       font=("Consolas", 9, "bold"))

        self.btn_start = tk.Button(body, text="▶  START SESSION",
                                   bg=SUCCESS, fg=BG,
                                   activebackground="#00ff87",
                                   activeforeground=BG,
                                   command=self._start_session, **btn_cfg)
        self.btn_start.pack(fill="x", pady=(0, 4))

        self.btn_stop = tk.Button(body, text="■  STOP SESSION",
                                  bg=ACCENT2, fg="white",
                                  activebackground="#ff8a80",
                                  activeforeground="white",
                                  state="disabled",
                                  command=self._stop_session, **btn_cfg)
        self.btn_stop.pack(fill="x", pady=(0, 4))

        self.btn_status = tk.Button(body, text="ℹ  CHECK STATUS",
                                    bg=PANEL, fg=ACCENT,
                                    activebackground=ACCENT,
                                    activeforeground=BG,
                                    command=self._check_status, **btn_cfg)
        self.btn_status.pack(fill="x", pady=(0, 4))

        self.btn_pause = tk.Button(body, text="⏸  PAUSE DISPLAY",
                                   bg=PANEL, fg=WARNING,
                                   activebackground=WARNING,
                                   activeforeground=BG,
                                   state="disabled",
                                   command=self._toggle_pause, **btn_cfg)
        self.btn_pause.pack(fill="x", pady=(0, 4))

        self.btn_shot = tk.Button(body, text="📷  SCREENSHOT",
                                  bg=PANEL, fg=TEXT,
                                  activebackground=BORDER,
                                  activeforeground=ACCENT,
                                  state="disabled",
                                  command=self._screenshot, **btn_cfg)
        self.btn_shot.pack(fill="x")

    # ── right panel ───────────────────────────────────────────────────────

    def _build_right(self, pane):
        frame = tk.Frame(pane, bg=BG)
        frame.rowconfigure(0, weight=3)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # ── video canvas ───────────────────────────────────────────────────
        vid_card = tk.Frame(frame, bg=PANEL, bd=0)
        vid_card.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        vid_card.rowconfigure(1, weight=1)
        vid_card.columnconfigure(0, weight=1)

        # header
        vid_hdr = tk.Frame(vid_card, bg=PANEL, height=32)
        vid_hdr.grid(row=0, column=0, sticky="ew")
        vid_hdr.pack_propagate(False)
        tk.Label(vid_hdr, text="LIVE FEED", fg=ACCENT, bg=PANEL,
                 font=("Consolas", 8, "bold")).pack(side="left", padx=10, pady=6)
        self.hud_label = tk.Label(vid_hdr, text="", fg=MUTED, bg=PANEL, font=FONT_MONO)
        self.hud_label.pack(side="right", padx=10)

        self.canvas = tk.Canvas(vid_card, bg="#080a0e", bd=0,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self._draw_placeholder()

        # ── event log ─────────────────────────────────────────────────────
        log_card = tk.Frame(frame, bg=PANEL, bd=0)
        log_card.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)

        log_hdr = tk.Frame(log_card, bg=PANEL, height=28)
        log_hdr.grid(row=0, column=0, sticky="ew")
        log_hdr.pack_propagate(False)
        tk.Label(log_hdr, text="EVENT LOG", fg=ACCENT, bg=PANEL,
                 font=("Consolas", 8, "bold")).pack(side="left", padx=10, pady=5)
        tk.Button(log_hdr, text="Clear", bg=PANEL, fg=MUTED,
                  activebackground=BORDER, activeforeground=TEXT,
                  relief="flat", cursor="hand2", font=FONT_LABEL,
                  command=self._clear_log).pack(side="right", padx=8, pady=4)

        self.log = scrolledtext.ScrolledText(
            log_card, bg="#080a0e", fg=TEXT,
            font=FONT_MONO, relief="flat", bd=0,
            state="disabled", wrap="word",
            insertbackground=ACCENT,
            selectbackground=BORDER)
        self.log.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))
        self.log.tag_config("alert",   foreground=ACCENT2)
        self.log.tag_config("info",    foreground=ACCENT)
        self.log.tag_config("success", foreground=SUCCESS)
        self.log.tag_config("warn",    foreground=WARNING)
        self.log.tag_config("muted",   foreground=MUTED)

        return frame

    # ── placeholder canvas drawing ────────────────────────────────────────

    def _draw_placeholder(self):
        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(), 640)
        h = max(self.canvas.winfo_height(), 360)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill="#080a0e", outline="")
        self.canvas.create_text(w//2, h//2 - 16,
                                text="⬡", fill=BORDER,
                                font=("Consolas", 48))
        self.canvas.create_text(w//2, h//2 + 36,
                                text="Awaiting stream...",
                                fill=MUTED, font=("Consolas", 11))

    # ── source radio logic ────────────────────────────────────────────────

    def _source_changed(self):
        src = self.var_src.get()
        # hide all context rows
        for row in (self.video_row, self.jfile_row, self.cam_row):
            row.grid_remove()
        if src == "video_file":
            self.video_row.grid()
        elif src == "jetson_file":
            self.jfile_row.grid()
        elif src == "client_camera":
            self.cam_row.grid()

    # ── file browsers ─────────────────────────────────────────────────────

    def _browse_video(self):
        p = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if p:
            self.var_video.set(p)

    def _browse_rule(self):
        p = filedialog.askopenfilename(
            title="Select Rule YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All", "*.*")])
        if p:
            self.var_rule.set(p)

    # ── dwell slider ──────────────────────────────────────────────────────

    def _dwell_changed(self, val):
        v = round(float(val), 1)
        self.dwell_label.config(text=f"{v}s")
        self.dwell_seconds = v

    # ── log helpers ───────────────────────────────────────────────────────

    def _log(self, msg, tag="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.config(state="normal")
        self.log.insert("end", f"[{ts}] {msg}\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    # ── status indicator ──────────────────────────────────────────────────

    def _set_status(self, label, color):
        self.status_dot.config(fg=color)
        self.status_label.config(text=label, fg=color)

    # ── session actions ───────────────────────────────────────────────────

    def _base_url(self):
        return f"http://{self.var_host.get()}:{self.var_port.get()}"

    def _ws_url(self):
        return f"ws://{self.var_host.get()}:{self.var_port.get()}/session/stream"

    def _start_session(self):
        if not REQUESTS_AVAILABLE:
            messagebox.showerror("Missing dep", "Install: pip install requests")
            return
        if not WS_AVAILABLE:
            messagebox.showerror("Missing dep", "Install: pip install websockets")
            return

        rule_path = self.var_rule.get().strip()
        if not rule_path:
            messagebox.showwarning("Rule YAML", "Please select a Rule YAML file.")
            return

        src = self.var_src.get()
        if src == "video_file" and not self.var_video.get().strip():
            messagebox.showwarning("Video", "Please select a local video file.")
            return

        self._log("Starting session...", "info")
        self._set_status("CONNECTING", WARNING)
        self.btn_start.config(state="disabled")

        def _run():
            try:
                base_url = self._base_url()
                rule_text = Path(rule_path).read_text(encoding="utf-8")
                rule_data = {}
                if YAML_AVAILABLE:
                    try:
                        rule_data = yaml.safe_load(rule_text) or {}
                    except Exception:
                        pass
                self.dwell_seconds = rule_data.get(
                    "conditions", {}).get("dwell_seconds", 2.0)
                self.after(0, lambda: self.var_dwell.set(self.dwell_seconds))
                self.after(0, lambda: self.dwell_label.config(
                    text=f"{self.dwell_seconds}s"))

                if src == "video_file":
                    jetson_path = upload_video(
                        base_url, self.var_video.get().strip(),
                        progress_cb=lambda m: self.after(0, lambda m=m: self._log(m, "info")))
                    source_type = "video_file"
                elif src == "jetson_file":
                    jetson_path = self.var_jfile.get().strip()
                    source_type = "jetson_file"
                else:
                    jetson_path = None
                    source_type = "client_camera"

                start_session(
                    base_url=base_url,
                    source_type=source_type,
                    rule_yaml_str=rule_text,
                    jetson_file_path=jetson_path,
                    detector=self.var_detector.get(),
                    device=self.var_device.get(),
                    det_interval=int(self.var_det_int.get()),
                    stream_width=int(self.var_sw.get()),
                    jpeg_quality=int(self.var_jq.get()),
                )
                self.after(0, lambda: self._log("Session started ✓", "success"))
                self.after(0, self._on_session_started)

                # reset queues
                self.stop_flag.clear()
                while not self.frame_q.empty():
                    self.frame_q.get_nowait()
                while not self.event_q.empty():
                    self.event_q.get_nowait()

                # launch WS
                if source_type == "client_camera":
                    asyncio.run(stream_camera(
                        self._ws_url(), int(self.var_cam.get()),
                        self.frame_q, self.event_q,
                        self.stop_flag, int(self.var_jq.get())))
                else:
                    asyncio.run(receive_stream(
                        self._ws_url(), self.frame_q, self.event_q,
                        self.stop_flag))
            except Exception as exc:
                self.after(0, lambda: self._log(f"ERROR: {exc}", "alert"))
                self.after(0, lambda: self._set_status("ERROR", ACCENT2))
                self.after(0, lambda: self.btn_start.config(state="normal"))

        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()

    def _on_session_started(self):
        self.session_running = True
        self._set_status("LIVE", SUCCESS)
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_pause.config(state="normal")
        self.btn_shot.config(state="normal")

    def _stop_session(self):
        self._log("Stopping session...", "warn")
        self.stop_flag.set()

        def _do_stop():
            try:
                stop_session(self._base_url())
                self.after(0, lambda: self._log("Session stopped.", "warn"))
            except Exception as e:
                self.after(0, lambda: self._log(f"Stop error: {e}", "alert"))
            finally:
                self.after(0, self._on_session_stopped)

        threading.Thread(target=_do_stop, daemon=True).start()

    def _on_session_stopped(self):
        self.session_running = False
        self._set_status("IDLE", MUTED)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_pause.config(state="disabled")
        self.btn_shot.config(state="disabled")
        self._draw_placeholder()

    def _check_status(self):
        def _do():
            try:
                result = check_status(self._base_url())
                self.after(0, lambda: self._log(
                    json.dumps(result, indent=2), "muted"))
            except Exception as e:
                self.after(0, lambda: self._log(f"Status error: {e}", "alert"))

        threading.Thread(target=_do, daemon=True).start()

    def _push_config(self):
        def _do():
            try:
                result = update_config(
                    self._base_url(),
                    dwell_seconds=round(self.var_dwell.get(), 1),
                    min_confidence=round(self.var_conf.get(), 3),
                    box_threshold=round(self.var_bth.get(), 3),
                )
                changed = result.get("changed", {})
                self.after(0, lambda: self._log(
                    f"Config updated: {changed}", "success"))
            except Exception as e:
                self.after(0, lambda: self._log(f"Config error: {e}", "alert"))

        threading.Thread(target=_do, daemon=True).start()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  RESUME DISPLAY")
            self._log("Display paused.", "warn")
        else:
            self.btn_pause.config(text="⏸  PAUSE DISPLAY")
            self._log("Display resumed.", "info")

    def _screenshot(self):
        if self.last_frame is not None:
            self.shot_n += 1
            fn = f"ovd_screenshot_{self.shot_n:03d}.jpg"
            cv2.imwrite(fn, self.last_frame)
            self._log(f"Screenshot saved → {fn}", "success")

    # ── frame polling ─────────────────────────────────────────────────────

    def _poll_frames(self):
        try:
            while not self.frame_q.empty():
                frame, meta = self.frame_q.get_nowait()
                if not self.paused:
                    self.last_frame = frame
                    vis = _draw_annotations(frame, meta)
                    self._render_frame(vis, meta)
        except queue.Empty:
            pass
        self.after(33, self._poll_frames)

    def _render_frame(self, frame, meta):
        if not PIL_AVAILABLE:
            return
        h_c = self.canvas.winfo_height()
        w_c = self.canvas.winfo_width()
        if h_c < 2 or w_c < 2:
            return

        h_f, w_f = frame.shape[:2]
        scale = min(w_c / w_f, h_c / h_f)
        nw, nh = int(w_f * scale), int(h_f * scale)
        resized = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w_c, h_c, fill="#080a0e", outline="")
        x0 = (w_c - nw) // 2
        y0 = (h_c - nh) // 2
        self.canvas.create_image(x0, y0, anchor="nw", image=self._tk_img)

        # HUD
        fps    = meta.get("fps", "--")
        tracks = meta.get("tracks", 0)
        events = meta.get("events", 0)
        fid    = meta.get("frame_id", 0)
        hud = f"FPS {fps}  |  Tracks {tracks}  |  Events {events}  |  Frame {fid}  |  dwell {self.dwell_seconds}s"
        self.hud_label.config(text=hud)

        if self.paused:
            self.canvas.create_text(w_c//2, h_c//2,
                                    text="⏸  PAUSED",
                                    fill=ACCENT, font=("Consolas", 22, "bold"))

    # ── event polling ─────────────────────────────────────────────────────

    def _poll_events(self):
        try:
            while not self.event_q.empty():
                evt = self.event_q.get_nowait()
                self._print_event(evt)
        except queue.Empty:
            pass
        self.after(200, self._poll_events)

    def _print_event(self, evt: dict):
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"{'─'*50}",
            f"  ⚠  ALERT  [{ts}]",
            f"  rule     : {evt.get('rule_id')}",
            f"  track_id : {evt.get('track_id')}",
            f"  camera   : {evt.get('camera_id')}",
            f"  action   : {evt.get('action')}",
            f"  at       : {evt.get('timestamp')}s  frame={evt.get('evidence',{}).get('frame_id')}",
            f"{'─'*50}",
        ]
        for line in lines:
            self._log(line, "alert")

    # ── close ─────────────────────────────────────────────────────────────

    def _on_close(self):
        if self.session_running:
            if messagebox.askyesno("Quit", "Stop the active session and quit?"):
                self.stop_flag.set()
                try:
                    stop_session(self._base_url())
                except Exception:
                    pass
                self.destroy()
        else:
            self.destroy()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not PIL_AVAILABLE:
        print("[WARN] Pillow not found — live video will not render.\n"
              "       pip install pillow")
    if not REQUESTS_AVAILABLE:
        print("[WARN] requests not found — install: pip install requests")
    if not WS_AVAILABLE:
        print("[WARN] websockets not found — install: pip install websockets")

    app = OVDClientGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
