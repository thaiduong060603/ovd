"""
gui_tkinter.py  —  OVD Watchdog GUI (Tkinter version)
Python 3.10 | No extra dependencies beyond opencv-python + Pillow

Install:
    pip install Pillow opencv-python pyyaml

Run:
    python gui_tkinter.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import time
import cv2
import yaml
import json
from pathlib import Path
from PIL import Image, ImageTk
from dataclasses import dataclass, field
from typing import Optional


# ── Color palette (dark industrial theme) ────────────────────────────────────
BG_DARK    = "#1a1a1a"
BG_PANEL   = "#242424"
BG_CARD    = "#2e2e2e"
ACCENT     = "#00c8ff"
ACCENT2    = "#ff4c4c"
TEXT_MAIN  = "#e8e8e8"
TEXT_DIM   = "#888888"
TEXT_GREEN = "#00e676"
TEXT_RED   = "#ff5252"
TEXT_YELLOW= "#ffd740"
BORDER     = "#3a3a3a"
FONT_MONO  = ("Courier New", 9)
FONT_UI    = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 8)


# ── Pipeline state ────────────────────────────────────────────────────────────
@dataclass
class PipelineArgs:
    input_source: str  = ""
    detector:     str  = "yolo_world"
    device:       str  = "auto"
    camera_id:    str  = "cam_01"
    det_interval: int  = 5
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    dwell_seconds: float = 2.0
    min_confidence: float = 0.40
    min_frames:   int  = 3
    rule_path:    str  = ""
    display:      bool = True
    output:       str  = ""


class StatusUpdate:
    def __init__(self, fps=0.0, tracks=0, incidents=0, events=0,
                 state="idle", log_line="", frame=None):
        self.fps       = fps
        self.tracks    = tracks
        self.incidents = incidents
        self.events    = events
        self.state     = state
        self.log_line  = log_line
        self.frame     = frame   # numpy BGR frame or None


# ── Main GUI ──────────────────────────────────────────────────────────────────
class OVDWatchdogApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OVD Watchdog")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1100, 680)

        self.args          = PipelineArgs()
        self.pipeline_thread: Optional[threading.Thread] = None
        self.stop_event    = threading.Event()
        self.status_queue  = queue.Queue(maxsize=10)
        self._photo_ref    = None   # keep ImageTk reference alive
        self._rule_data    = {}     # raw loaded yaml dict

        # String vars for linked widgets
        self._sv = {}
        self._build_ui()
        self._apply_dark_style()
        self._start_status_poll()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top title bar
        title_bar = tk.Frame(self, bg=BG_DARK, height=48)
        title_bar.pack(fill="x", padx=0, pady=0)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="⬡  OVD WATCHDOG", bg=BG_DARK,
                 fg=ACCENT, font=("Courier New", 13, "bold")).pack(side="left", padx=18, pady=10)
        self._state_label = tk.Label(title_bar, text="● IDLE", bg=BG_DARK,
                                     fg=TEXT_DIM, font=FONT_MONO)
        self._state_label.pack(side="right", padx=18)

        # Main body
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left — video panel
        self._build_video_panel(body)

        # Right — controls
        right = tk.Frame(body, bg=BG_DARK, width=370)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        tab_src  = tk.Frame(notebook, bg=BG_PANEL)
        tab_rule = tk.Frame(notebook, bg=BG_PANEL)
        tab_args = tk.Frame(notebook, bg=BG_PANEL)
        tab_log  = tk.Frame(notebook, bg=BG_PANEL)

        notebook.add(tab_src,  text=" Source ")
        notebook.add(tab_rule, text=" Rule ")
        notebook.add(tab_args, text=" Args ")
        notebook.add(tab_log,  text=" Log ")

        self._build_source_tab(tab_src)
        self._build_rule_tab(tab_rule)
        self._build_args_tab(tab_args)
        self._build_log_tab(tab_log)

        # Bottom — status bar + pipeline controls
        self._build_bottom_bar(right)

    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.pack(side="left", fill="both", expand=True)

        header = tk.Frame(frame, bg=BG_CARD, height=32)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="LIVE FEED", bg=BG_CARD,
                 fg=TEXT_DIM, font=FONT_SMALL).pack(side="left", padx=10, pady=6)
        self._fps_label = tk.Label(header, text="FPS: --", bg=BG_CARD,
                                   fg=ACCENT, font=FONT_MONO)
        self._fps_label.pack(side="right", padx=10)

        self._canvas = tk.Canvas(frame, bg="#0d0d0d", bd=0, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas_size = (640, 480)

        # Placeholder text
        self._canvas.create_text(320, 240, text="No feed — start pipeline",
                                 fill=TEXT_DIM, font=("Courier New", 11),
                                 tags="placeholder")

        # Stats overlay
        stats = tk.Frame(frame, bg=BG_CARD, height=28)
        stats.pack(fill="x")
        stats.pack_propagate(False)
        self._track_lbl   = self._stat_label(stats, "Tracks: 0")
        self._incident_lbl= self._stat_label(stats, "Incidents: 0")
        self._event_lbl   = self._stat_label(stats, "Events: 0")

    def _stat_label(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=BG_CARD, fg=TEXT_DIM, font=FONT_SMALL)
        lbl.pack(side="left", padx=12, pady=4)
        return lbl

    def _build_source_tab(self, parent):
        p = tk.Frame(parent, bg=BG_PANEL)
        p.pack(fill="both", expand=True, padx=12, pady=12)

        self._section(p, "INPUT SOURCE")

        # Radio: camera vs file
        self._sv["source_type"] = tk.StringVar(value="file")
        rb_frame = tk.Frame(p, bg=BG_PANEL)
        rb_frame.pack(fill="x", pady=(4, 8))
        tk.Radiobutton(rb_frame, text="Video file", variable=self._sv["source_type"],
                       value="file", bg=BG_PANEL, fg=TEXT_MAIN, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, font=FONT_UI,
                       command=self._toggle_source).pack(side="left", padx=4)
        tk.Radiobutton(rb_frame, text="Camera", variable=self._sv["source_type"],
                       value="camera", bg=BG_PANEL, fg=TEXT_MAIN, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, font=FONT_UI,
                       command=self._toggle_source).pack(side="left", padx=12)

        # File path
        self._file_frame = tk.Frame(p, bg=BG_PANEL)
        self._file_frame.pack(fill="x", pady=2)
        self._sv["input_path"] = tk.StringVar()
        tk.Label(self._file_frame, text="Path:", bg=BG_PANEL,
                 fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
        row = tk.Frame(self._file_frame, bg=BG_PANEL)
        row.pack(fill="x")
        self._path_entry = tk.Entry(row, textvariable=self._sv["input_path"],
                                    bg=BG_CARD, fg=TEXT_MAIN, insertbackground=ACCENT,
                                    relief="flat", font=FONT_UI, bd=4)
        self._path_entry.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse", bg=BG_CARD, fg=ACCENT, relief="flat",
                  font=FONT_SMALL, cursor="hand2",
                  command=self._browse_video).pack(side="right", padx=(4, 0))

        # Camera index
        self._cam_frame = tk.Frame(p, bg=BG_PANEL)
        tk.Label(self._cam_frame, text="Camera index:", bg=BG_PANEL,
                 fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
        self._sv["cam_index"] = tk.StringVar(value="0")
        tk.Entry(self._cam_frame, textvariable=self._sv["cam_index"],
                 bg=BG_CARD, fg=TEXT_MAIN, insertbackground=ACCENT,
                 relief="flat", font=FONT_UI, bd=4, width=8).pack(anchor="w")

        self._section(p, "DETECTOR")
        self._sv["detector"] = tk.StringVar(value="yolo_world")
        for val, lbl in [("yolo_world", "YOLO-World"), ("groundingdino", "GroundingDINO")]:
            tk.Radiobutton(p, text=lbl, variable=self._sv["detector"],
                           value=val, bg=BG_PANEL, fg=TEXT_MAIN, selectcolor=BG_CARD,
                           activebackground=BG_PANEL, font=FONT_UI).pack(anchor="w", padx=4)

        self._section(p, "DEVICE")
        self._sv["device"] = tk.StringVar(value="auto")
        for val in ["auto", "cpu", "cuda", "mps"]:
            tk.Radiobutton(p, text=val, variable=self._sv["device"],
                           value=val, bg=BG_PANEL, fg=TEXT_MAIN, selectcolor=BG_CARD,
                           activebackground=BG_PANEL, font=FONT_UI).pack(anchor="w", padx=4)

        self._sv["camera_id"] = tk.StringVar(value="cam_01")
        self._labeled_entry(p, "Camera ID", self._sv["camera_id"])

    def _build_rule_tab(self, parent):
        p = tk.Frame(parent, bg=BG_PANEL)
        p.pack(fill="both", expand=True, padx=12, pady=12)

        self._section(p, "RULE FILE")
        row = tk.Frame(p, bg=BG_PANEL)
        row.pack(fill="x", pady=(4, 0))
        self._sv["rule_path"] = tk.StringVar()
        tk.Entry(row, textvariable=self._sv["rule_path"],
                 bg=BG_CARD, fg=TEXT_MAIN, insertbackground=ACCENT,
                 relief="flat", font=FONT_UI, bd=4).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Load", bg=ACCENT, fg=BG_DARK, relief="flat",
                  font=("Segoe UI", 9, "bold"), cursor="hand2",
                  command=self._load_rule).pack(side="right", padx=(4, 0))

        self._rule_loaded_lbl = tk.Label(p, text="No rule loaded", bg=BG_PANEL,
                                         fg=TEXT_DIM, font=FONT_SMALL)
        self._rule_loaded_lbl.pack(anchor="w", pady=(2, 8))

        self._section(p, "RULE CONDITIONS")
        self._sv["dwell_seconds"]  = tk.StringVar(value="2.0")
        self._sv["min_confidence"] = tk.StringVar(value="0.40")
        self._sv["min_frames"]     = tk.StringVar(value="3")
        self._sv["cooldown"]       = tk.StringVar(value="30")

        self._labeled_entry(p, "dwell_seconds",  self._sv["dwell_seconds"])
        self._labeled_entry(p, "min_confidence", self._sv["min_confidence"])
        self._labeled_entry(p, "min_frames",     self._sv["min_frames"])
        self._labeled_entry(p, "cooldown_seconds", self._sv["cooldown"])

        self._section(p, "DETECTION THRESHOLDS")
        self._sv["box_threshold"]  = tk.StringVar(value="0.30")
        self._sv["text_threshold"] = tk.StringVar(value="0.25")
        self._labeled_entry(p, "box_threshold",  self._sv["box_threshold"])
        self._labeled_entry(p, "text_threshold", self._sv["text_threshold"])

        self._section(p, "RULE PROMPT")
        self._sv["prompt_positive"] = tk.StringVar(value="person . worker .")
        self._labeled_entry(p, "prompt_positive", self._sv["prompt_positive"])

        tk.Button(p, text="Apply changes to rule", bg=BG_CARD, fg=ACCENT,
                  relief="flat", font=FONT_UI, cursor="hand2",
                  command=self._apply_rule_changes).pack(fill="x", pady=(12, 0))

    def _build_args_tab(self, parent):
        p = tk.Frame(parent, bg=BG_PANEL)
        p.pack(fill="both", expand=True, padx=12, pady=12)

        self._section(p, "PIPELINE ARGS")
        self._sv["det_interval"] = tk.StringVar(value="5")
        self._labeled_entry(p, "detection_interval (frames)", self._sv["det_interval"])

        self._sv["output_path"] = tk.StringVar(value="")
        self._labeled_entry(p, "output video path (optional)", self._sv["output_path"])

        self._section(p, "DETECTION INTERVAL PRESETS")
        presets = tk.Frame(p, bg=BG_PANEL)
        presets.pack(fill="x", pady=4)
        for val, lbl in [("1","Every frame"),("5","x5 (default)"),("10","x10"),("30","x30 (sparse)")]:
            tk.Button(presets, text=lbl, bg=BG_CARD, fg=TEXT_MAIN,
                      relief="flat", font=FONT_SMALL, cursor="hand2",
                      command=lambda v=val: self._sv["det_interval"].set(v)
                      ).pack(side="left", padx=2, pady=2)

        self._section(p, "TRACKER SETTINGS")
        self._sv["track_thresh"]  = tk.StringVar(value="0.4")
        self._sv["track_buffer"]  = tk.StringVar(value="90")
        self._sv["match_thresh"]  = tk.StringVar(value="0.5")
        self._labeled_entry(p, "track_thresh",  self._sv["track_thresh"])
        self._labeled_entry(p, "track_buffer",  self._sv["track_buffer"])
        self._labeled_entry(p, "match_thresh",  self._sv["match_thresh"])

        info = tk.Label(p,
            text="Changes take effect on next pipeline start.",
            bg=BG_PANEL, fg=TEXT_DIM, font=FONT_SMALL, wraplength=320, justify="left")
        info.pack(anchor="w", pady=(16, 0))

    def _build_log_tab(self, parent):
        p = tk.Frame(parent, bg=BG_PANEL)
        p.pack(fill="both", expand=True)
        self._log_box = scrolledtext.ScrolledText(
            p, bg="#0d0d0d", fg=TEXT_MAIN, insertbackground=ACCENT,
            relief="flat", font=FONT_MONO, state="disabled", wrap="word"
        )
        self._log_box.pack(fill="both", expand=True, padx=4, pady=4)
        # Color tags
        self._log_box.tag_config("detect",  foreground=ACCENT)
        self._log_box.tag_config("event",   foreground=TEXT_GREEN)
        self._log_box.tag_config("error",   foreground=TEXT_RED)
        self._log_box.tag_config("warn",    foreground=TEXT_YELLOW)
        self._log_box.tag_config("dim",     foreground=TEXT_DIM)

        btn_row = tk.Frame(p, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=4, pady=(0, 4))
        tk.Button(btn_row, text="Clear log", bg=BG_CARD, fg=TEXT_DIM,
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._clear_log).pack(side="right")

    def _build_bottom_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_DARK, height=56)
        bar.pack(fill="x", pady=(6, 0))
        bar.pack_propagate(False)

        self._start_btn = tk.Button(
            bar, text="▶  START", bg=TEXT_GREEN, fg=BG_DARK,
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
            command=self._start_pipeline, padx=16
        )
        self._start_btn.pack(side="left", padx=(8, 4), pady=8)

        self._stop_btn = tk.Button(
            bar, text="■  STOP", bg=ACCENT2, fg="white",
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
            command=self._stop_pipeline, padx=16, state="disabled"
        )
        self._stop_btn.pack(side="left", padx=4, pady=8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG_PANEL, fg=TEXT_DIM,
                 font=FONT_SMALL).pack(anchor="w", pady=(12, 2))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 6))

    def _labeled_entry(self, parent, label, sv):
        tk.Label(parent, text=label, bg=BG_PANEL,
                 fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w", pady=(4, 0))
        tk.Entry(parent, textvariable=sv,
                 bg=BG_CARD, fg=TEXT_MAIN, insertbackground=ACCENT,
                 relief="flat", font=FONT_UI, bd=4).pack(fill="x", pady=(0, 2))

    def _apply_dark_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",       background=BG_PANEL, borderwidth=0)
        style.configure("TNotebook.Tab",   background=BG_CARD, foreground=TEXT_DIM,
                         font=FONT_UI, padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)])

    def _on_canvas_resize(self, event):
        self._canvas_size = (event.width, event.height)

    # ── Source toggle ─────────────────────────────────────────────────────────

    def _toggle_source(self):
        if self._sv["source_type"].get() == "file":
            self._cam_frame.pack_forget()
            self._file_frame.pack(fill="x", pady=2)
        else:
            self._file_frame.pack_forget()
            self._cam_frame.pack(fill="x", pady=2)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if path:
            self._sv["input_path"].set(path)

    # ── Rule load / apply ─────────────────────────────────────────────────────

    def _load_rule(self):
        path = filedialog.askopenfilename(
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            self._rule_data = data
            self._sv["rule_path"].set(path)
            # Populate fields from loaded rule
            cond = data.get("conditions", {})
            det  = data.get("detection", {})
            act  = data.get("actions", {})
            self._sv["dwell_seconds"].set(str(cond.get("dwell_seconds", 2.0)))
            self._sv["min_confidence"].set(str(cond.get("min_confidence", 0.40)))
            self._sv["min_frames"].set(str(cond.get("min_frames", 3)))
            self._sv["cooldown"].set(str(act.get("cooldown_seconds", 30)))
            self._sv["box_threshold"].set(str(det.get("box_threshold", 0.30)))
            self._sv["text_threshold"].set(str(det.get("text_threshold", 0.25)))
            self._sv["prompt_positive"].set(det.get("prompt_positive", ""))
            rule_id = data.get("rule_id", "unknown")
            self._rule_loaded_lbl.config(text=f"✓ Loaded: {rule_id}", fg=TEXT_GREEN)
            self._log(f"Rule loaded: {Path(path).name} [{rule_id}]", "event")
        except Exception as e:
            messagebox.showerror("Rule load error", str(e))
            self._log(f"Rule load error: {e}", "error")

    def _apply_rule_changes(self):
        """Push GUI values back into _rule_data and save as temp file."""
        if not self._rule_data:
            messagebox.showwarning("No rule", "Load a rule file first.")
            return
        try:
            self._rule_data.setdefault("conditions", {})
            self._rule_data.setdefault("detection", {})
            self._rule_data.setdefault("actions", {})
            self._rule_data["conditions"]["dwell_seconds"]  = float(self._sv["dwell_seconds"].get())
            self._rule_data["conditions"]["min_confidence"] = float(self._sv["min_confidence"].get())
            self._rule_data["conditions"]["min_frames"]     = int(self._sv["min_frames"].get())
            self._rule_data["actions"]["cooldown_seconds"]  = float(self._sv["cooldown"].get())
            self._rule_data["detection"]["box_threshold"]   = float(self._sv["box_threshold"].get())
            self._rule_data["detection"]["text_threshold"]  = float(self._sv["text_threshold"].get())
            self._rule_data["detection"]["prompt_positive"] = self._sv["prompt_positive"].get()
            # Write temp rule file
            tmp = Path("configs/rules/_gui_active_rule.yaml")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                yaml.dump(self._rule_data, f, default_flow_style=False, allow_unicode=True)
            self._sv["rule_path"].set(str(tmp))
            self._log(f"Rule saved → {tmp}", "event")
        except Exception as e:
            messagebox.showerror("Apply error", str(e))

    # ── Pipeline control ──────────────────────────────────────────────────────

    def _collect_args(self) -> Optional[PipelineArgs]:
        a = PipelineArgs()
        try:
            if self._sv["source_type"].get() == "file":
                a.input_source = self._sv["input_path"].get().strip()
            else:
                a.input_source = int(self._sv["cam_index"].get())
            a.rule_path     = self._sv["rule_path"].get().strip()
            a.detector      = self._sv["detector"].get()
            a.device        = self._sv["device"].get()
            a.camera_id     = self._sv["camera_id"].get().strip()
            a.det_interval  = int(self._sv["det_interval"].get())
            a.box_threshold = float(self._sv["box_threshold"].get())
            a.text_threshold= float(self._sv["text_threshold"].get())
            a.dwell_seconds = float(self._sv["dwell_seconds"].get())
            a.min_confidence= float(self._sv["min_confidence"].get())
            a.min_frames    = int(self._sv["min_frames"].get())
            a.output        = self._sv["output_path"].get().strip()
        except ValueError as e:
            messagebox.showerror("Invalid args", str(e))
            return None
        if not a.input_source:
            messagebox.showwarning("No input", "Select a video file or camera.")
            return None
        if not a.rule_path:
            messagebox.showwarning("No rule", "Load a rule YAML file.")
            return None
        return a

    def _start_pipeline(self):
        args = self._collect_args()
        if args is None:
            return
        self.args = args
        self.stop_event.clear()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_state("RUNNING", TEXT_GREEN)
        self._canvas.delete("placeholder")
        self._log("── Pipeline starting ──", "event")
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_worker, args=(args,), daemon=True
        )
        self.pipeline_thread.start()

    def _stop_pipeline(self):
        self.stop_event.set()
        self._log("Stop requested…", "warn")

    def _pipeline_worker(self, args: PipelineArgs):
        """Runs in background thread — drives the actual pipeline."""
        try:
            # ── Lazy imports (so GUI starts fast) ────────────────────────────
            import sys, os
            sys.path.insert(0, os.getcwd())

            from src.core.ingest.video_source import create_video_source
            from src.core.track.byte_tracker import ByteTracker
            from src.core.rules.rule_engine_core_v1 import RuleEngineV1
            from src.models.rule import Rule
            ### duonglt
            from src.utils.visualization import Visualizer

            rule       = Rule.from_yaml(args.rule_path)
            video_src  = create_video_source(args.input_source)
            tracker    = ByteTracker(track_thresh=0.4, track_buffer=90,
                                     match_thresh=0.5, frame_rate=video_src.fps)
            engine     = RuleEngineV1(rules=[rule], camera_id=args.camera_id)

            visualizer = Visualizer()

            # detector
            if args.detector == "yolo_world":
                from src.core.detect.yolo_world_detector import YOLOWorldDetector
                detector = YOLOWorldDetector(
                    model_path="models/yolov8s-world.pt",
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    device=args.device,
                )
            else:
                from src.core.detect.grounding_dino_detector import GroundingDINODetector
                detector = GroundingDINODetector(
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    device=args.device,
                )

            self._push_status(StatusUpdate(state="running", log_line="Pipeline started ✓"))

            frame_id        = 0
            det_cache       = []
            last_det_frame  = -999
            fps_start       = time.time()
            fps_counter     = 0
            current_fps     = 0.0
            emitted_ids     = set()

            while not self.stop_event.is_set():
                ok, frame = video_src.read()
                if not ok:
                    self._push_status(StatusUpdate(state="ended",
                                                   log_line="Video ended."))
                    break
                frame_id  += 1
                timestamp  = frame_id / video_src.fps

                if frame_id % args.det_interval == 0:
                    det_cache      = detector.detect(frame, [rule.prompt_positive], frame_id)
                    last_det_frame = frame_id
                    if det_cache:
                        self._push_status(StatusUpdate(
                            log_line=f"[DETECT] frame={frame_id} → {len(det_cache)} obj(s)",
                        ))

                stale     = frame_id - last_det_frame
                det_input = det_cache if stale < args.det_interval else []
                tracks    = tracker.update(det_input, frame_id, timestamp)
                incidents = engine.evaluate(tracks, frame_id, timestamp)

                # New events
                new_evts = [e for e in engine.events if e.event_id not in emitted_ids]
                for evt in new_evts:
                    emitted_ids.add(evt.event_id)
                    self._push_status(StatusUpdate(
                        log_line=f"[EVENT] {evt.event_id}  rule={evt.rule_id}  track={evt.track_id}",
                    ))

                # FPS
                fps_counter += 1
                if fps_counter >= 15:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_start   = time.time()
                    fps_counter = 0

                # Draw bboxes on frame copy
                vis = frame.copy()
                # for t in tracks:
                #     x1,y1,x2,y2 = map(int, t.bbox)
                #     color = (0,200,255) if t.state == "confirmed" else (120,120,120)
                #     cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                #     cv2.putText(vis, f"#{t.track_id} {t.class_name[:6]}",
                #                 (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # ROI
                if rule.roi and rule.roi.enabled:
                    vis = visualizer.draw_roi(vis, rule.roi)

                # Tracks
                vis = visualizer.draw_tracks(
                    vis,
                    tracks,
                    show_id=True,
                    show_state=True
                )

                # Incidents
                vis = visualizer.draw_incidents(
                    vis,
                    engine.get_active_incidents(),
                    tracks
                )

                # Info panel
                vis = visualizer.add_info_panel_with_incidents(
                    vis,
                    frame_id,
                    current_fps,
                    len(det_cache),
                    len(tracks),
                    len(engine.get_active_incidents()),
                    len(engine.get_confirmed_incidents())
                )                

                self._push_status(StatusUpdate(
                    fps=current_fps,
                    tracks=len(tracks),
                    incidents=len(engine.get_active_incidents()),
                    events=len(engine.events),
                    state="running",
                    frame=vis,
                ))

            video_src.release()

        except Exception as e:
            import traceback
            self._push_status(StatusUpdate(state="error",
                                           log_line=f"ERROR: {e}\n{traceback.format_exc()}"))
        finally:
            self._push_status(StatusUpdate(state="stopped",
                                           log_line="── Pipeline stopped ──"))

    def _push_status(self, s: StatusUpdate):
        try:
            self.status_queue.put_nowait(s)
        except queue.Full:
            pass

    # ── Status poll (main thread) ─────────────────────────────────────────────

    def _start_status_poll(self):
        self._poll_status()

    def _poll_status(self):
        try:
            while True:
                s = self.status_queue.get_nowait()
                if s.log_line:
                    tag = ("event" if "[EVENT]" in s.log_line
                           else "detect" if "[DETECT]" in s.log_line
                           else "error" if "ERROR" in s.log_line
                           else "warn" if "stop" in s.log_line.lower()
                           else "dim")
                    self._log(s.log_line, tag)
                if s.fps:
                    self._fps_label.config(text=f"FPS: {s.fps:.1f}")
                if s.tracks is not None:
                    self._track_lbl.config(text=f"Tracks: {s.tracks}")
                if s.incidents is not None:
                    self._incident_lbl.config(text=f"Incidents: {s.incidents}")
                if s.events is not None:
                    self._event_lbl.config(text=f"Events: {s.events}")
                if s.state:
                    self._update_pipeline_state(s.state)
                if s.frame is not None:
                    self._display_frame(s.frame)
        except queue.Empty:
            pass
        self.after(33, self._poll_status)  # ~30 fps UI refresh

    def _update_pipeline_state(self, state: str):
        mapping = {
            "running": ("● RUNNING", TEXT_GREEN),
            "stopped": ("● STOPPED", TEXT_DIM),
            "ended":   ("● ENDED",   TEXT_YELLOW),
            "error":   ("● ERROR",   TEXT_RED),
            "idle":    ("● IDLE",    TEXT_DIM),
        }
        text, color = mapping.get(state, ("● " + state.upper(), TEXT_DIM))
        self._state_label.config(text=text, fg=color)
        if state in ("stopped", "ended", "error"):
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            # Clear video canvas
            self._canvas.delete("all")
            self._canvas.create_text(
                self._canvas_size[0]//2, self._canvas_size[1]//2,
                text="Feed stopped", fill=TEXT_DIM,
                font=("Courier New", 11), tags="placeholder"
            )

    def _set_state(self, label, color):
        self._state_label.config(text=f"● {label}", fg=color)

    def _display_frame(self, frame_bgr):
        cw, ch = self._canvas_size
        if cw < 10 or ch < 10:
            return
        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)
        photo   = ImageTk.PhotoImage(image=img)
        self._photo_ref = photo
        self._canvas.delete("all")
        ox, oy  = (cw - nw) // 2, (ch - nh) // 2
        self._canvas.create_image(ox, oy, anchor="nw", image=photo)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str = "dim"):
        ts = time.strftime("%H:%M:%S")
        self._log_box.config(state="normal")
        self._log_box.insert("end", f"[{ts}] {text}\n", tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.stop_event.set()
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=2.0)
        self.destroy()


if __name__ == "__main__":
    app = OVDWatchdogApp()
    app.mainloop()
