"""
gui_dearpygui.py  —  OVD Watchdog GUI (Dear PyGui version)
Python 3.10 | Renders via OpenGL — runs well on Jetson

Install:
    pip install dearpygui opencv-python pyyaml

Run:
    python gui_dearpygui.py
"""

import dearpygui.dearpygui as dpg
import threading
import queue
import time
import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ── Theme constants ────────────────────────────────────────────────────────────
W_MAIN   = 1280
H_MAIN   = 760
W_VIDEO  = 820
W_PANEL  = 440

C_BG          = (18,  18,  20,  255)
C_PANEL       = (28,  28,  32,  255)
C_CARD        = (38,  38,  44,  255)
C_ACCENT      = (0,   200, 255, 255)
C_GREEN       = (0,   230, 118, 255)
C_RED         = (255, 75,  75,  255)
C_YELLOW      = (255, 215, 64,  255)
C_TEXT        = (230, 230, 235, 255)
C_DIM         = (130, 130, 140, 255)
C_BORDER      = (55,  55,  62,  255)


# ── Pipeline state ─────────────────────────────────────────────────────────────
@dataclass
class StatusUpdate:
    fps:       float = 0.0
    tracks:    int   = 0
    incidents: int   = 0
    events:    int   = 0
    state:     str   = ""
    log_line:  str   = ""
    frame:     object = None   # numpy BGR or None


# ── App ────────────────────────────────────────────────────────────────────────
class OVDApp:
    def __init__(self):
        self.stop_event    = threading.Event()
        self.status_queue  = queue.Queue(maxsize=20)
        self.pipeline_thread: Optional[threading.Thread] = None
        self._rule_data    = {}
        self._emitted_ids  = set()
        self._texture_data = None
        self._vid_w        = 640
        self._vid_h        = 480
        self._frame_ready  = False

    def run(self):
        dpg.create_context()
        self._register_texture()
        self._build_theme()
        self._build_ui()
        dpg.create_viewport(title="OVD Watchdog", width=W_MAIN, height=H_MAIN,
                            resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        while dpg.is_dearpygui_running():
            self._poll_status()
            dpg.render_dearpygui_frame()

        self.stop_event.set()
        dpg.destroy_context()

    # ── Texture ────────────────────────────────────────────────────────────────

    def _register_texture(self):
        self._vid_w = 640
        self._vid_h = 480
        blank = np.zeros((self._vid_h, self._vid_w, 4), dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=self._vid_w, height=self._vid_h,
                default_value=blank.flatten().tolist(),
                format=dpg.mvFormat_Float_rgba,
                tag="video_texture"
            )

    def _update_texture(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if w != self._vid_w or h != self._vid_h:
            # Re-register texture at new size
            self._vid_w, self._vid_h = w, h
            if dpg.does_item_exist("video_texture"):
                dpg.delete_item("video_texture")
            blank = np.zeros((h, w, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=w, height=h,
                    default_value=blank.flatten().tolist(),
                    format=dpg.mvFormat_Float_rgba,
                    tag="video_texture"
                )
            if dpg.does_item_exist("video_image"):
                dpg.configure_item("video_image", texture_tag="video_texture",
                                   width=w, height=h)

        rgba  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        data  = (rgba / 255.0).astype(np.float32).flatten().tolist()
        dpg.set_value("video_texture", data)

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _build_theme(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        C_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50,50,58,255))
                dpg.add_theme_color(dpg.mvThemeCol_Button,         C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (55,55,65,255))
                dpg.add_theme_color(dpg.mvThemeCol_Header,         C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (50,50,58,255))
                dpg.add_theme_color(dpg.mvThemeCol_Tab,            C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered,     (55,55,65,255))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive,      C_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  C_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_Text,           C_TEXT)
                dpg.add_theme_color(dpg.mvThemeCol_Border,         C_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        C_PANEL)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 10)
        dpg.bind_theme(global_theme)

        # Green button theme
        with dpg.theme() as self._green_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (0, 160, 80, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 190, 100, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (10, 10, 10, 255))

        # Red button theme
        with dpg.theme() as self._red_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (180, 40, 40, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 60, 60, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255, 255))

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        with dpg.window(tag="main_window", label="", no_title_bar=True,
                        no_move=True, no_resize=False):

            # ── Title bar ────────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("⬡  OVD WATCHDOG", color=C_ACCENT)
                dpg.add_spacer(width=W_MAIN - 280)
                dpg.add_text("State:", color=C_DIM)
                dpg.add_text("IDLE", tag="state_label", color=C_DIM)

            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── Body: video (left) + controls (right) ─────────────────────────
            with dpg.group(horizontal=True):

                # ── LEFT: video panel ─────────────────────────────────────────
                with dpg.child_window(width=W_VIDEO, height=H_MAIN - 80,
                                      border=True, tag="video_panel"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("LIVE FEED", color=C_DIM)
                        dpg.add_spacer(width=W_VIDEO - 200)
                        dpg.add_text("FPS: --", tag="fps_label", color=C_ACCENT)

                    dpg.add_image("video_texture", width=W_VIDEO - 20,
                                  height=H_MAIN - 160, tag="video_image")

                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text("Tracks:", color=C_DIM)
                        dpg.add_text("0", tag="tracks_val", color=C_TEXT)
                        dpg.add_spacer(width=20)
                        dpg.add_text("Incidents:", color=C_DIM)
                        dpg.add_text("0", tag="incidents_val", color=C_TEXT)
                        dpg.add_spacer(width=20)
                        dpg.add_text("Events:", color=C_DIM)
                        dpg.add_text("0", tag="events_val", color=C_TEXT)

                # ── RIGHT: control panel ──────────────────────────────────────
                with dpg.child_window(width=W_PANEL, height=H_MAIN - 80,
                                      border=True):

                    with dpg.tab_bar():

                        # ── Tab: Source ───────────────────────────────────────
                        with dpg.tab(label=" Source "):
                            dpg.add_spacer(height=6)
                            self._section("INPUT SOURCE")

                            dpg.add_text("Type:", color=C_DIM)
                            dpg.add_radio_button(
                                items=["Video file", "Camera"],
                                tag="source_type", horizontal=True, default_value=0
                            )
                            dpg.add_spacer(height=4)

                            dpg.add_text("Path:", color=C_DIM)
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(tag="input_path", width=W_PANEL - 100,
                                                   hint="path/to/video.mp4")
                                dpg.add_button(label="Browse",
                                               callback=self._browse_video)
                            dpg.add_input_text(label="Camera index", tag="cam_index",
                                               default_value="0", width=80)

                            self._section("DETECTOR")
                            dpg.add_radio_button(
                                items=["yolo_world", "groundingdino"],
                                tag="detector", horizontal=True, default_value=0
                            )

                            self._section("DEVICE")
                            dpg.add_radio_button(
                                items=["auto", "cpu", "cuda", "mps"],
                                tag="device", horizontal=True, default_value=0
                            )

                            self._section("CAMERA ID")
                            dpg.add_input_text(tag="camera_id",
                                               default_value="cam_01", width=160)

                        # ── Tab: Rule ─────────────────────────────────────────
                        with dpg.tab(label=" Rule "):
                            dpg.add_spacer(height=6)
                            self._section("RULE FILE")
                            with dpg.group(horizontal=True):
                                dpg.add_input_text(tag="rule_path",
                                                   width=W_PANEL - 100,
                                                   hint="configs/rules/rule.yaml")
                                dpg.add_button(label="Load",
                                               callback=self._load_rule)
                            dpg.add_text("No rule loaded", tag="rule_status",
                                         color=C_DIM)

                            self._section("CONDITIONS")
                            dpg.add_input_float(label="dwell_seconds",
                                                tag="dwell_seconds",
                                                default_value=2.0,
                                                min_value=0.1, max_value=60.0,
                                                width=120)
                            dpg.add_input_float(label="min_confidence",
                                                tag="min_confidence",
                                                default_value=0.40,
                                                min_value=0.0, max_value=1.0,
                                                width=120)
                            dpg.add_input_int(label="min_frames",
                                              tag="min_frames",
                                              default_value=3,
                                              min_value=1, max_value=30,
                                              width=120)
                            dpg.add_input_float(label="cooldown_seconds",
                                                tag="cooldown",
                                                default_value=30.0,
                                                min_value=1.0, max_value=3600.0,
                                                width=120)

                            self._section("DETECTION THRESHOLDS")
                            dpg.add_input_float(label="box_threshold",
                                                tag="box_threshold",
                                                default_value=0.30,
                                                min_value=0.0, max_value=1.0,
                                                width=120)
                            dpg.add_input_float(label="text_threshold",
                                                tag="text_threshold",
                                                default_value=0.25,
                                                min_value=0.0, max_value=1.0,
                                                width=120)

                            self._section("PROMPT")
                            dpg.add_input_text(label="prompt_positive",
                                               tag="prompt_positive",
                                               default_value="person . worker .",
                                               width=W_PANEL - 60)
                            dpg.add_spacer(height=8)
                            dpg.add_button(label="Apply rule changes",
                                           width=W_PANEL - 30,
                                           callback=self._apply_rule_changes)

                        # ── Tab: Args ─────────────────────────────────────────
                        with dpg.tab(label=" Args "):
                            dpg.add_spacer(height=6)
                            self._section("PIPELINE ARGS")
                            dpg.add_input_int(label="detection_interval",
                                              tag="det_interval",
                                              default_value=5,
                                              min_value=1, max_value=60,
                                              width=120)

                            self._section("INTERVAL PRESETS")
                            with dpg.group(horizontal=True):
                                for val, lbl in [(1,"x1"),(5,"x5"),(10,"x10"),(30,"x30")]:
                                    dpg.add_button(
                                        label=lbl,
                                        callback=lambda s, a, v=val: dpg.set_value(
                                            "det_interval", v)
                                    )

                            self._section("TRACKER SETTINGS")
                            dpg.add_input_float(label="track_thresh",
                                                tag="track_thresh",
                                                default_value=0.4,
                                                min_value=0.0, max_value=1.0,
                                                width=120)
                            dpg.add_input_int(label="track_buffer",
                                              tag="track_buffer",
                                              default_value=90,
                                              min_value=10, max_value=300,
                                              width=120)
                            dpg.add_input_float(label="match_thresh",
                                                tag="match_thresh",
                                                default_value=0.5,
                                                min_value=0.0, max_value=1.0,
                                                width=120)
                            dpg.add_input_text(label="output video (optional)",
                                               tag="output_path",
                                               default_value="",
                                               width=W_PANEL - 60)
                            dpg.add_spacer(height=8)
                            dpg.add_text("Changes apply on next pipeline start.",
                                         color=C_DIM)

                        # ── Tab: Log ──────────────────────────────────────────
                        with dpg.tab(label=" Log "):
                            dpg.add_spacer(height=6)
                            dpg.add_input_text(
                                tag="log_box",
                                multiline=True,
                                readonly=True,
                                width=W_PANEL - 20,
                                height=H_MAIN - 180,
                                default_value=""
                            )
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Clear",
                                               callback=lambda: dpg.set_value("log_box", ""))

                    # ── Pipeline controls ─────────────────────────────────────
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="▶  START", tag="start_btn",
                                       width=180, height=36,
                                       callback=self._start_pipeline)
                        dpg.bind_item_theme("start_btn", self._green_btn_theme)

                        dpg.add_spacer(width=8)

                        dpg.add_button(label="■  STOP", tag="stop_btn",
                                       width=180, height=36,
                                       callback=self._stop_pipeline,
                                       enabled=False)
                        dpg.bind_item_theme("stop_btn", self._red_btn_theme)

    def _section(self, title: str):
        dpg.add_spacer(height=4)
        dpg.add_text(title, color=C_DIM)
        dpg.add_separator()
        dpg.add_spacer(height=2)

    # ── File dialogs ───────────────────────────────────────────────────────────

    def _browse_video(self):
        # Dear PyGui file dialog
        with dpg.file_dialog(
            directory_selector=False, show=True,
            callback=self._on_video_selected,
            file_count=1, tag="video_file_dialog",
            width=600, height=400
        ):
            dpg.add_file_extension(".mp4")
            dpg.add_file_extension(".avi")
            dpg.add_file_extension(".mov")
            dpg.add_file_extension(".mkv")

    def _on_video_selected(self, sender, app_data):
        if app_data["selections"]:
            path = list(app_data["selections"].values())[0]
            dpg.set_value("input_path", path)

    # ── Rule load / apply ──────────────────────────────────────────────────────

    def _load_rule(self):
        with dpg.file_dialog(
            directory_selector=False, show=True,
            callback=self._on_rule_selected,
            tag="rule_file_dialog", width=600, height=400
        ):
            dpg.add_file_extension(".yaml")
            dpg.add_file_extension(".yml")

    def _on_rule_selected(self, sender, app_data):
        if not app_data["selections"]:
            return
        path = list(app_data["selections"].values())[0]
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            self._rule_data = data
            dpg.set_value("rule_path", path)
            cond = data.get("conditions", {})
            det  = data.get("detection", {})
            act  = data.get("actions", {})
            dpg.set_value("dwell_seconds",  cond.get("dwell_seconds", 2.0))
            dpg.set_value("min_confidence", cond.get("min_confidence", 0.40))
            dpg.set_value("min_frames",     cond.get("min_frames", 3))
            dpg.set_value("cooldown",       act.get("cooldown_seconds", 30.0))
            dpg.set_value("box_threshold",  det.get("box_threshold", 0.30))
            dpg.set_value("text_threshold", det.get("text_threshold", 0.25))
            dpg.set_value("prompt_positive",det.get("prompt_positive", ""))
            rule_id = data.get("rule_id", "unknown")
            dpg.set_value("rule_status", f"✓ Loaded: {rule_id}")
            dpg.configure_item("rule_status", color=C_GREEN)
            self._log(f"Rule loaded: {Path(path).name} [{rule_id}]", "EVENT")
        except Exception as e:
            dpg.set_value("rule_status", f"Error: {e}")
            dpg.configure_item("rule_status", color=C_RED)

    def _apply_rule_changes(self):
        if not self._rule_data:
            return
        try:
            self._rule_data.setdefault("conditions", {})
            self._rule_data.setdefault("detection", {})
            self._rule_data.setdefault("actions", {})
            self._rule_data["conditions"]["dwell_seconds"]  = dpg.get_value("dwell_seconds")
            self._rule_data["conditions"]["min_confidence"] = dpg.get_value("min_confidence")
            self._rule_data["conditions"]["min_frames"]     = dpg.get_value("min_frames")
            self._rule_data["actions"]["cooldown_seconds"]  = dpg.get_value("cooldown")
            self._rule_data["detection"]["box_threshold"]   = dpg.get_value("box_threshold")
            self._rule_data["detection"]["text_threshold"]  = dpg.get_value("text_threshold")
            self._rule_data["detection"]["prompt_positive"] = dpg.get_value("prompt_positive")
            tmp = Path("configs/rules/_gui_active_rule.yaml")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                yaml.dump(self._rule_data, f, default_flow_style=False, allow_unicode=True)
            dpg.set_value("rule_path", str(tmp))
            self._log(f"Rule saved → {tmp}", "EVENT")
        except Exception as e:
            self._log(f"Apply error: {e}", "ERROR")

    # ── Pipeline control ───────────────────────────────────────────────────────

    def _start_pipeline(self):
        src_type   = dpg.get_value("source_type")
        input_src  = (dpg.get_value("input_path").strip()
                      if src_type == "Video file"
                      else int(dpg.get_value("cam_index")))
        rule_path  = dpg.get_value("rule_path").strip()

        if not input_src:
            self._log("ERROR: no input source selected", "ERROR")
            return
        if not rule_path:
            self._log("ERROR: no rule file loaded", "ERROR")
            return

        self.stop_event.clear()
        self._emitted_ids.clear()
        dpg.configure_item("start_btn", enabled=False)
        dpg.configure_item("stop_btn",  enabled=True)
        self._set_state("RUNNING", C_GREEN)
        self._log("── Pipeline starting ──", "EVENT")

        args = dict(
            input_source   = input_src,
            rule_path      = rule_path,
            detector       = dpg.get_value("detector"),
            device         = dpg.get_value("device"),
            camera_id      = dpg.get_value("camera_id"),
            det_interval   = dpg.get_value("det_interval"),
            box_threshold  = dpg.get_value("box_threshold"),
            text_threshold = dpg.get_value("text_threshold"),
            dwell_seconds  = dpg.get_value("dwell_seconds"),
            min_confidence = dpg.get_value("min_confidence"),
            min_frames     = dpg.get_value("min_frames"),
            output         = dpg.get_value("output_path"),
        )
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_worker, kwargs=args, daemon=True
        )
        self.pipeline_thread.start()

    def _stop_pipeline(self):
        self.stop_event.set()
        self._log("Stop requested…", "WARN")

    def _pipeline_worker(self, input_source, rule_path, detector, device,
                         camera_id, det_interval, box_threshold, text_threshold,
                         dwell_seconds, min_confidence, min_frames, output):
        try:
            import sys, os
            sys.path.insert(0, os.getcwd())

            from src.core.ingest.video_source import create_video_source
            from src.core.track.byte_tracker import ByteTracker
            from src.core.rules.rule_engine_core_v1 import RuleEngineV1
            from src.models.rule import Rule

            rule      = Rule.from_yaml(rule_path)
            video_src = create_video_source(input_source)
            tracker   = ByteTracker(track_thresh=0.4, track_buffer=90,
                                    match_thresh=0.5, frame_rate=video_src.fps)
            engine    = RuleEngineV1(rules=[rule], camera_id=camera_id)

            if detector == "yolo_world":
                from src.core.detect.yolo_world_detector import YOLOWorldDetector
                det = YOLOWorldDetector(
                    model_path="models/yolov8s-world.pt",
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                )
            else:
                from src.core.detect.grounding_dino_detector import GroundingDINODetector
                det = GroundingDINODetector(
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                )

            self._push(StatusUpdate(state="running",
                                    log_line="Pipeline started ✓"))

            frame_id       = 0
            det_cache      = []
            last_det_frame = -999
            t_fps          = time.time()
            fps_n          = 0
            cur_fps        = 0.0

            # Resize frame to fit video panel
            target_w = W_VIDEO - 24
            target_h = H_MAIN - 160

            while not self.stop_event.is_set():
                ok, frame = video_src.read()
                if not ok:
                    self._push(StatusUpdate(state="ended",
                                            log_line="Video ended."))
                    break

                frame_id  += 1
                timestamp  = frame_id / video_src.fps

                if frame_id % det_interval == 0:
                    det_cache      = det.detect(frame, [rule.prompt_positive], frame_id)
                    last_det_frame = frame_id
                    if det_cache:
                        self._push(StatusUpdate(
                            log_line=f"[DETECT] frame={frame_id} → {len(det_cache)} obj(s)"
                        ))

                stale     = frame_id - last_det_frame
                det_input = det_cache if stale < det_interval else []
                tracks    = tracker.update(det_input, frame_id, timestamp)
                incidents = engine.evaluate(tracks, frame_id, timestamp)

                # Events
                new_evts = [e for e in engine.events
                            if e.event_id not in self._emitted_ids]
                for evt in new_evts:
                    self._emitted_ids.add(evt.event_id)
                    self._push(StatusUpdate(
                        log_line=f"[EVENT] {evt.event_id} rule={evt.rule_id} track={evt.track_id}"
                    ))

                # FPS
                fps_n += 1
                if fps_n >= 15:
                    cur_fps = fps_n / (time.time() - t_fps)
                    t_fps   = time.time()
                    fps_n   = 0

                # Draw
                vis = frame.copy()
                for t in tracks:
                    x1,y1,x2,y2 = map(int, t.bbox)
                    col = (0,200,255) if t.state == "confirmed" else (100,100,110)
                    cv2.rectangle(vis, (x1,y1),(x2,y2), col, 2)
                    cv2.putText(vis, f"#{t.track_id}", (x1, max(y1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # Resize to panel size
                h_f, w_f = vis.shape[:2]
                scale = min(target_w / w_f, target_h / h_f)
                vis   = cv2.resize(vis, (int(w_f*scale), int(h_f*scale)))

                self._push(StatusUpdate(
                    fps=cur_fps,
                    tracks=len(tracks),
                    incidents=len(engine.get_active_incidents()),
                    events=len(engine.events),
                    state="running",
                    frame=vis,
                ))

            video_src.release()

        except Exception as e:
            import traceback
            self._push(StatusUpdate(state="error",
                                    log_line=f"ERROR: {e}\n{traceback.format_exc()}"))
        finally:
            self._push(StatusUpdate(state="stopped",
                                    log_line="── Pipeline stopped ──"))

    def _push(self, s: StatusUpdate):
        try:
            self.status_queue.put_nowait(s)
        except queue.Full:
            pass

    # ── Status poll (called every frame from main loop) ───────────────────────

    def _poll_status(self):
        try:
            while True:
                s = self.status_queue.get_nowait()
                if s.log_line:
                    tag = ("[EVENT]" if "EVENT" in s.log_line
                           else "[DETECT]" if "DETECT" in s.log_line
                           else "[ERROR]" if "ERROR" in s.log_line
                           else "")
                    self._log(s.log_line, tag)
                if s.fps:
                    dpg.set_value("fps_label", f"FPS: {s.fps:.1f}")
                if s.tracks is not None:
                    dpg.set_value("tracks_val", str(s.tracks))
                if s.incidents is not None:
                    dpg.set_value("incidents_val", str(s.incidents))
                if s.events is not None:
                    dpg.set_value("events_val", str(s.events))
                if s.state:
                    self._update_state(s.state)
                if s.frame is not None:
                    self._update_texture(s.frame)
        except queue.Empty:
            pass

    def _update_state(self, state: str):
        mapping = {
            "running": ("RUNNING", C_GREEN),
            "stopped": ("STOPPED", C_DIM),
            "ended":   ("ENDED",   C_YELLOW),
            "error":   ("ERROR",   C_RED),
            "idle":    ("IDLE",    C_DIM),
        }
        text, color = mapping.get(state, (state.upper(), C_DIM))
        dpg.set_value("state_label", text)
        dpg.configure_item("state_label", color=color)
        if state in ("stopped", "ended", "error"):
            dpg.configure_item("start_btn", enabled=True)
            dpg.configure_item("stop_btn",  enabled=False)

    def _set_state(self, text, color):
        dpg.set_value("state_label", text)
        dpg.configure_item("state_label", color=color)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str = ""):
        ts  = time.strftime("%H:%M:%S")
        cur = dpg.get_value("log_box")
        line = f"[{ts}] {text}\n"
        # Keep log from growing too large
        lines = (cur + line).split("\n")
        if len(lines) > 300:
            lines = lines[-300:]
        dpg.set_value("log_box", "\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = OVDApp()
    app.run()
