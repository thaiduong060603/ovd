"""
ovd_web_app.py  —  OVD Watchdog Web Application
Flask-based web frontend that proxies all communication to the Jetson GPU server.

Architecture:
    Browser ←→ Flask Web App (this file) ←→ Jetson Server (jetson_server.py)

Install:
    pip install flask flask-socketio requests websockets pyyaml opencv-python

Run:
    python ovd_web_app.py
    python ovd_web_app.py --host 0.0.0.0 --port 5000 --jetson http://localhost:8765

    If running ALL-IN-ONE on Jetson (recommended):
    python ovd_web_app.py --jetson http://localhost:8765

Endpoints (this Flask app):
    GET  /                    → Main dashboard UI
    POST /api/start           → Start session (proxied to Jetson)
    POST /api/stop            → Stop session
    POST /api/config/update   → Hot-reload config
    POST /api/upload_video    → Upload video (proxied to Jetson)
    GET  /api/status          → Session status
    GET  /api/events          → Polling events
    WS   /ws/stream           → SocketIO relay of Jetson WebSocket stream
"""

import argparse
import asyncio
import base64
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import websockets
from flask import Flask, jsonify, render_template_string, request, Response
from flask_socketio import SocketIO, emit

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

JETSON_BASE_URL = "http://localhost:8765"   # overridable via --jetson arg

app = Flask(__name__)
app.config["SECRET_KEY"] = "ovd-watchdog-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024)

# Internal relay state
_relay_stop   = threading.Event()
_relay_thread = None


# ─────────────────────────────────────────────────────────────────────────────
# Jetson proxy helpers
# ─────────────────────────────────────────────────────────────────────────────

def jetson_get(path, **kwargs):
    return requests.get(f"{JETSON_BASE_URL}{path}", timeout=10, **kwargs)

def jetson_post(path, **kwargs):
    return requests.post(f"{JETSON_BASE_URL}{path}", timeout=60, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket relay: Jetson → SocketIO → Browser
# ─────────────────────────────────────────────────────────────────────────────

def _relay_worker(ws_url: str, stop_flag: threading.Event):
    """Background thread: connects to Jetson WS and relays frames/events to browser via SocketIO."""
    reconnect_delay = 2.0

    async def _run():
        nonlocal reconnect_delay
        while not stop_flag.is_set():
            try:
                async with websockets.connect(
                    ws_url,
                    max_size=20 * 1024 * 1024,
                    ping_interval=20,
                ) as ws:
                    reconnect_delay = 2.0
                    async for raw in ws:
                        if stop_flag.is_set():
                            break
                        if isinstance(raw, bytes):
                            if raw.startswith(b"FRAME"):
                                idx = raw.find(b"__META__")
                                if idx == -1:
                                    continue
                                jpeg_bytes = raw[5:idx]
                                meta_bytes = raw[idx + 8:]
                                try:
                                    meta = json.loads(meta_bytes.decode("utf-8"))
                                except Exception:
                                    meta = {}
                                # Encode JPEG as base64 for browser
                                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                                socketio.emit("frame", {"jpeg": b64, "meta": meta})

                            elif raw.startswith(b"EVIDENCE"):
                                idx = raw.find(b"__META__")
                                if idx == -1:
                                    continue
                                jpeg_bytes = raw[8:idx]
                                meta_bytes = raw[idx + 8:]
                                try:
                                    ev_meta = json.loads(meta_bytes.decode("utf-8"))
                                except Exception:
                                    ev_meta = {}
                                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                                socketio.emit("evidence", {"jpeg": b64, "meta": ev_meta})

                        elif isinstance(raw, str):
                            try:
                                msg = json.loads(raw)
                                if msg.get("type") == "event":
                                    socketio.emit("alert_event", msg.get("payload", {}))
                                elif msg.get("type") == "status":
                                    socketio.emit("status_update", msg.get("payload", {}))
                            except Exception:
                                pass
            except (ConnectionRefusedError, OSError):
                pass
            except Exception:
                pass

            if not stop_flag.is_set():
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 15.0)

    asyncio.run(_run())


def _camera_relay_worker(ws_url: str, stop_flag: threading.Event, cam_index: int = 0, jpeg_quality: int = 70):
    """Background thread: reads local camera, sends to Jetson, relays results to browser."""
    async def _run():
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            socketio.emit("log", {"msg": f"Cannot open camera index {cam_index}", "level": "error"})
            return
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        try:
            async with websockets.connect(ws_url, max_size=20 * 1024 * 1024, ping_interval=20) as ws:
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
                        if isinstance(raw, bytes):
                            if raw.startswith(b"FRAME"):
                                idx = raw.find(b"__META__")
                                if idx == -1:
                                    continue
                                jpeg_bytes = raw[5:idx]
                                meta_bytes = raw[idx + 8:]
                                try:
                                    meta = json.loads(meta_bytes.decode("utf-8"))
                                except Exception:
                                    meta = {}
                                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                                socketio.emit("frame", {"jpeg": b64, "meta": meta})
                            elif raw.startswith(b"EVIDENCE"):
                                idx = raw.find(b"__META__")
                                if idx == -1:
                                    continue
                                jpeg_bytes = raw[8:idx]
                                meta_bytes = raw[idx + 8:]
                                try:
                                    ev_meta = json.loads(meta_bytes.decode("utf-8"))
                                except Exception:
                                    ev_meta = {}
                                b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                                socketio.emit("evidence", {"jpeg": b64, "meta": ev_meta})
                        elif isinstance(raw, str):
                            try:
                                msg = json.loads(raw)
                                if msg.get("type") == "event":
                                    socketio.emit("alert_event", msg.get("payload", {}))
                            except Exception:
                                pass

                await asyncio.gather(send_frames(), recv_results())
        finally:
            cap.release()

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# REST API routes (proxy to Jetson)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    try:
        r = jetson_get("/session/status")
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e), "state": "unreachable"}), 503


@app.route("/api/events")
def api_events():
    since = request.args.get("since_index", 0, type=int)
    try:
        r = jetson_get("/session/events", params={"since_index": since})
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    global _relay_stop, _relay_thread
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    try:
        r = requests.post(
            f"{JETSON_BASE_URL}/session/upload_video",
            files={"file": (f.filename, f.stream, f.content_type)},
            timeout=300,
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def api_start():
    global _relay_stop, _relay_thread

    data = request.get_json(force=True)
    source_type = data.get("source_type", "video_file")

    # Stop any existing relay
    _relay_stop.set()
    if _relay_thread and _relay_thread.is_alive():
        _relay_thread.join(timeout=3.0)
    _relay_stop = threading.Event()

    try:
        r = jetson_post("/session/start", json=data)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ws_url = JETSON_BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/session/stream"

    cam_index    = data.get("cam_index", 0)
    jpeg_quality = data.get("jpeg_quality", 70)

    if source_type == "client_camera":
        _relay_thread = threading.Thread(
            target=_camera_relay_worker,
            args=(ws_url, _relay_stop, cam_index, jpeg_quality),
            daemon=True,
        )
    else:
        _relay_thread = threading.Thread(
            target=_relay_worker,
            args=(ws_url, _relay_stop),
            daemon=True,
        )

    _relay_thread.start()
    return jsonify(r.json())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _relay_stop
    _relay_stop.set()
    try:
        r = jetson_post("/session/stop")
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/update", methods=["POST"])
def api_config_update():
    data = request.get_json(force=True)
    try:
        r = jetson_post("/session/config/update", json=data)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO events
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("log", {"msg": "Connected to OVD Watchdog Web Server", "level": "info"})


@socketio.on("ping_server")
def on_ping():
    emit("pong_server", {"time": time.time()})


# ─────────────────────────────────────────────────────────────────────────────
# Main HTML dashboard (single-page app)
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OVD Watchdog — Factory Safety Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── CSS VARIABLES ─────────────────────────────────────────────── */
:root {
  --bg:       #080b10;
  --surface:  #0e1219;
  --panel:    #111520;
  --border:   #1c2235;
  --border2:  #252d42;
  --accent:   #00d4ff;
  --accent2:  #ff3860;
  --accent3:  #00e676;
  --warning:  #ffb300;
  --text:     #c8d4f0;
  --muted:    #4a5578;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Syne', sans-serif;
}

/* ── RESET & BASE ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--mono); font-size: 13px; }

/* ── LAYOUT ────────────────────────────────────────────────────── */
.app        { display: flex; flex-direction: column; height: 100vh; }
.topbar     { display: flex; align-items: center; height: 52px; background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 20px; gap: 16px; flex-shrink: 0; }
.main       { display: flex; flex: 1; overflow: hidden; }
.sidebar    { width: 320px; min-width: 280px; background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }
.content    { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.video-area { flex: 1; position: relative; background: #05080d; overflow: hidden; }
.log-area   { height: 180px; flex-shrink: 0; background: var(--surface); border-top: 1px solid var(--border); }

/* ── TOPBAR ────────────────────────────────────────────────────── */
.logo { font-family: var(--sans); font-weight: 800; font-size: 15px; color: var(--accent); letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px; }
.logo svg { width: 22px; height: 22px; }
.topbar-spacer { flex: 1; }

.status-pill { display: flex; align-items: center; gap: 6px; background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 4px 12px; font-size: 11px; letter-spacing: 0.08em; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); transition: background 0.3s; }
.status-dot.live    { background: var(--accent3); box-shadow: 0 0 8px var(--accent3); animation: pulse 1.5s infinite; }
.status-dot.error   { background: var(--accent2); }
.status-dot.loading { background: var(--warning); animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

.hud-strip { display: flex; gap: 20px; font-size: 11px; color: var(--muted); }
.hud-item  { display: flex; flex-direction: column; align-items: center; }
.hud-value { font-size: 14px; font-weight: 700; color: var(--text); line-height: 1; }
.hud-label { font-size: 9px; letter-spacing: 0.1em; margin-top: 2px; }

/* ── SIDEBAR SECTIONS ──────────────────────────────────────────── */
.sidebar-section { border-bottom: 1px solid var(--border); }
.section-header  { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px 8px; cursor: pointer; user-select: none; }
.section-title   { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: var(--accent); }
.section-body    { padding: 0 14px 12px; }
.section-body.collapsed { display: none; }

/* ── FORM ELEMENTS ─────────────────────────────────────────────── */
.field       { margin-bottom: 8px; }
.field label { display: block; font-size: 10px; color: var(--muted); letter-spacing: 0.08em; margin-bottom: 3px; }
.field input, .field select, .field textarea {
  width: 100%; background: var(--surface); border: 1px solid var(--border2); color: var(--text);
  font-family: var(--mono); font-size: 12px; padding: 6px 8px; border-radius: 4px; outline: none;
  transition: border-color 0.2s;
}
.field input:focus, .field select:focus, .field textarea:focus { border-color: var(--accent); }
.field textarea { min-height: 90px; resize: vertical; }
.field select option { background: var(--surface); }

.field-row { display: flex; gap: 8px; }
.field-row .field { flex: 1; }

.radio-group { display: flex; flex-direction: column; gap: 4px; }
.radio-item  { display: flex; align-items: flex-start; gap: 8px; padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer; transition: border-color 0.2s, background 0.2s; }
.radio-item:hover   { border-color: var(--border2); background: var(--surface); }
.radio-item.active  { border-color: var(--accent); background: rgba(0,212,255,0.06); }
.radio-item input   { margin-top: 2px; accent-color: var(--accent); }
.radio-item-text    { flex: 1; }
.radio-item-name    { font-size: 12px; font-weight: 500; }
.radio-item-desc    { font-size: 10px; color: var(--muted); margin-top: 1px; }

.slider-row  { display: flex; align-items: center; gap: 8px; }
.slider-val  { min-width: 36px; text-align: right; color: var(--accent); font-weight: 700; }
input[type=range] { flex: 1; accent-color: var(--accent); }

/* ── BUTTONS ───────────────────────────────────────────────────── */
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 7px 14px; border-radius: 4px; font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; cursor: pointer; border: 1px solid transparent; transition: all 0.15s; white-space: nowrap; }
.btn:disabled { opacity: 0.35; cursor: not-allowed; }
.btn-primary { background: var(--accent); color: #000; border-color: var(--accent); }
.btn-primary:hover:not(:disabled) { background: #33ddff; }
.btn-danger  { background: transparent; color: var(--accent2); border-color: var(--accent2); }
.btn-danger:hover:not(:disabled) { background: rgba(255,56,96,0.12); }
.btn-ghost   { background: transparent; color: var(--muted); border-color: var(--border2); }
.btn-ghost:hover:not(:disabled) { color: var(--text); border-color: var(--border2); background: var(--surface); }
.btn-success { background: transparent; color: var(--accent3); border-color: var(--accent3); }
.btn-success:hover:not(:disabled) { background: rgba(0,230,118,0.1); }
.btn-warn    { background: transparent; color: var(--warning); border-color: var(--warning); }
.btn-warn:hover:not(:disabled) { background: rgba(255,179,0,0.1); }
.btn-full    { width: 100%; justify-content: center; }

.btn-group   { display: flex; gap: 6px; flex-wrap: wrap; }
.btn-group-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }

/* ── VIDEO CANVAS ──────────────────────────────────────────────── */
#videoCanvas { display: block; width: 100%; height: 100%; object-fit: contain; cursor: crosshair; }
.video-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none; }
#roiCanvas     { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

.video-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; opacity: 0.3; }
.placeholder-icon { font-size: 52px; line-height: 1; }
.placeholder-text { font-size: 13px; letter-spacing: 0.1em; }
.video-hud { position: absolute; bottom: 10px; left: 10px; background: rgba(8,11,16,0.75); border: 1px solid var(--border); border-radius: 4px; padding: 5px 10px; font-size: 10px; color: var(--muted); pointer-events: none; }

/* ── EVENT LOG ─────────────────────────────────────────────────── */
.log-header  { display: flex; align-items: center; justify-content: space-between; padding: 6px 14px; border-bottom: 1px solid var(--border); }
.log-title   { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: var(--accent); }
.log-body    { height: calc(100% - 33px); overflow-y: auto; padding: 4px 0; font-size: 11px; }
.log-entry   { display: flex; gap: 8px; padding: 2px 14px; line-height: 1.5; }
.log-ts      { color: var(--muted); flex-shrink: 0; }
.log-msg     { word-break: break-all; }
.log-msg.info    { color: var(--text); }
.log-msg.success { color: var(--accent3); }
.log-msg.warn    { color: var(--warning); }
.log-msg.error   { color: var(--accent2); }
.log-msg.alert   { color: var(--accent2); font-weight: 700; }
.log-msg.muted   { color: var(--muted); }

/* ── EVIDENCE PANEL ────────────────────────────────────────────── */
.evidence-strip { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.evidence-thumb { position: relative; width: 72px; height: 48px; border: 1px solid var(--border2); border-radius: 3px; overflow: hidden; cursor: pointer; flex-shrink: 0; }
.evidence-thumb img { width: 100%; height: 100%; object-fit: cover; }
.evidence-thumb .ev-badge { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(255,56,96,0.85); font-size: 8px; text-align: center; padding: 1px; color: #fff; }

/* ── MODAL ─────────────────────────────────────────────────────── */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; display: none; }
.modal-overlay.open { display: flex; }
.modal-box { background: var(--panel); border: 1px solid var(--border2); border-radius: 8px; max-width: 640px; width: 90%; max-height: 80vh; overflow: auto; }
.modal-img { width: 100%; display: block; }
.modal-meta { padding: 12px 16px; font-size: 11px; color: var(--muted); }
.modal-close { float: right; font-size: 18px; cursor: pointer; color: var(--muted); padding: 8px 16px; }

/* ── UPLOAD PROGRESS ───────────────────────────────────────────── */
.progress-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 6px; display: none; }
.progress-bar.active { display: block; }
.progress-fill { height: 100%; background: var(--accent); width: 0%; transition: width 0.3s; }

/* ── SCROLLBAR ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── ROI INFO ──────────────────────────────────────────────────── */
.roi-info { font-size: 10px; color: var(--muted); margin-top: 4px; }
.roi-pts  { color: var(--accent); }

/* ── RESPONSIVE ────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .sidebar { width: 100%; min-width: unset; }
  .main { flex-direction: column; }
  .hud-strip { display: none; }
}
</style>
</head>
<body>
<div class="app">

  <!-- ── TOP BAR ──────────────────────────────────────────────────── -->
  <div class="topbar">
    <div class="logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5"/>
        <circle cx="12" cy="12" r="3"/>
      </svg>
      OVD WATCHDOG
    </div>

    <div class="hud-strip" id="hudStrip">
      <div class="hud-item"><span class="hud-value" id="hudFps">—</span><span class="hud-label">FPS</span></div>
      <div class="hud-item"><span class="hud-value" id="hudTracks">—</span><span class="hud-label">TRACKS</span></div>
      <div class="hud-item"><span class="hud-value" id="hudEvents">—</span><span class="hud-label">EVENTS</span></div>
      <div class="hud-item"><span class="hud-value" id="hudFrame">—</span><span class="hud-label">FRAME</span></div>
    </div>

    <div class="topbar-spacer"></div>
    <div class="status-pill">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusLabel">IDLE</span>
    </div>
  </div>

  <!-- ── MAIN AREA ─────────────────────────────────────────────────── -->
  <div class="main">

    <!-- ── SIDEBAR ─────────────────────────────────────────────────── -->
    <div class="sidebar">

      <!-- CONNECTION -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">CONNECTION</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="field-row">
            <div class="field">
              <label>JETSON HOST</label>
              <input type="text" id="jetsonHost" value="localhost" placeholder="192.168.1.x"/>
            </div>
            <div class="field" style="max-width:80px">
              <label>PORT</label>
              <input type="text" id="jetsonPort" value="5000"/>
            </div>
          </div>
          <button class="btn btn-ghost btn-full" onclick="checkStatus()">⟳ CHECK SERVER</button>
        </div>
      </div>

      <!-- TASK TYPE -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">MONITORING TASK</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="radio-group" id="taskGroup">
            <label class="radio-item active">
              <input type="radio" name="task" value="1" checked onchange="onTaskChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">ROI Entry</div>
                <div class="radio-item-desc">Alert immediately upon zone entry</div>
              </div>
            </label>
            <label class="radio-item">
              <input type="radio" name="task" value="2" onchange="onTaskChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Dwell Time</div>
                <div class="radio-item-desc">Alert after staying in zone for X seconds</div>
              </div>
            </label>
            <label class="radio-item">
              <input type="radio" name="task" value="3" onchange="onTaskChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Helmet Safety</div>
                <div class="radio-item-desc">Detect workers missing helmets</div>
              </div>
            </label>
            <label class="radio-item">
              <input type="radio" name="task" value="4" onchange="onTaskChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Proximity Danger</div>
                <div class="radio-item-desc">Alert when objects are too close</div>
              </div>
            </label>
          </div>
        </div>
      </div>

      <!-- RULE YAML -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">RULE CONFIG (YAML)</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="field">
            <label>RULE YAML</label>
            <textarea id="ruleYaml" spellcheck="false">rule_id: rule_roi_entry
description: ROI Zone Entry Detection
detection:
  prompt_positive: person
  box_threshold: 0.25
  text_threshold: 0.25
conditions:
  dwell_seconds: 0
  min_confidence: 0.4
  min_frames: 3
actions:
  cooldown_seconds: 5
roi:
  enabled: true
  type: polygon
  points: []
</textarea>
          </div>
          <button class="btn btn-ghost btn-full" onclick="loadDefaultYaml()">⎙ LOAD TASK TEMPLATE</button>
        </div>
      </div>

      <!-- SOURCE -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">VIDEO SOURCE</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="radio-group" style="margin-bottom:10px" id="srcGroup">
            <label class="radio-item active">
              <input type="radio" name="src" value="video_file" checked onchange="onSrcChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Upload Video File</div>
              </div>
            </label>
            <label class="radio-item">
              <input type="radio" name="src" value="jetson_file" onchange="onSrcChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Jetson File Path</div>
              </div>
            </label>
            <label class="radio-item">
              <input type="radio" name="src" value="client_camera" onchange="onSrcChange(this)"/>
              <div class="radio-item-text">
                <div class="radio-item-name">Webcam (this device)</div>
              </div>
            </label>
          </div>

          <!-- video_file -->
          <div id="srcVideoFile">
            <div class="field">
              <label>VIDEO FILE</label>
              <input type="file" id="videoFile" accept="video/*" style="font-size:11px"/>
            </div>
            <div class="progress-bar" id="uploadProgress"><div class="progress-fill" id="uploadFill"></div></div>
            <button class="btn btn-ghost btn-full" id="btnUpload" onclick="uploadVideo()" style="margin-top:6px">⬆ UPLOAD TO JETSON</button>
          </div>

          <!-- jetson_file -->
          <div id="srcJetsonFile" style="display:none">
            <div class="field">
              <label>FILE PATH ON JETSON</label>
              <input type="text" id="jetsonFilePath" value="/app/videos/" placeholder="/path/to/video.mp4"/>
            </div>
          </div>

          <!-- client_camera -->
          <div id="srcCamera" style="display:none">
            <div class="field">
              <label>CAMERA INDEX</label>
              <input type="number" id="camIndex" value="0" min="0" max="10"/>
            </div>
          </div>
        </div>
      </div>

      <!-- PIPELINE PARAMS -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">PIPELINE PARAMS</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="field-row">
            <div class="field">
              <label>DETECTOR</label>
              <select id="detector">
                <option value="yolo_world">YOLO-World</option>
                <option value="grounding_dino">GroundingDINO</option>
              </select>
            </div>
            <div class="field">
              <label>DEVICE</label>
              <select id="device">
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label>DETECTION INTERVAL <span class="slider-val" id="detIntVal">5</span></label>
            <input type="range" id="detInterval" min="1" max="30" value="5" oninput="document.getElementById('detIntVal').textContent=this.value"/>
          </div>
          <div class="field">
            <label>STREAM WIDTH <span class="slider-val" id="streamWidthVal">640</span></label>
            <input type="range" id="streamWidth" min="320" max="1280" step="64" value="640" oninput="document.getElementById('streamWidthVal').textContent=this.value"/>
          </div>
          <div class="field">
            <label>JPEG QUALITY <span class="slider-val" id="jpegQVal">75</span></label>
            <input type="range" id="jpegQuality" min="30" max="95" step="5" value="75" oninput="document.getElementById('jpegQVal').textContent=this.value"/>
          </div>
        </div>
      </div>

      <!-- SESSION CONTROLS -->
      <div class="sidebar-section">
        <div class="section-header"><span class="section-title">SESSION</span></div>
        <div class="section-body">
          <div class="btn-group-2">
            <button class="btn btn-primary" id="btnStart" onclick="startSession()">▶ START</button>
            <button class="btn btn-danger" id="btnStop"  onclick="stopSession()" disabled>■ STOP</button>
          </div>
        </div>
      </div>

      <!-- LIVE CONTROLS -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">LIVE CONTROLS</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <div class="field">
            <label>DWELL TIME <span class="slider-val" id="dwellVal">2.0</span>s</label>
            <input type="range" id="dwellSlider" min="0" max="30" step="0.5" value="2" oninput="document.getElementById('dwellVal').textContent=parseFloat(this.value).toFixed(1)"/>
          </div>
          <div class="field">
            <label>CONFIDENCE THRESHOLD <span class="slider-val" id="confVal">0.40</span></label>
            <input type="range" id="confSlider" min="0.1" max="0.95" step="0.05" value="0.4" oninput="document.getElementById('confVal').textContent=parseFloat(this.value).toFixed(2)"/>
          </div>
          <div class="field">
            <label>BOX THRESHOLD <span class="slider-val" id="bthVal">0.25</span></label>
            <input type="range" id="bthSlider" min="0.05" max="0.9" step="0.05" value="0.25" oninput="document.getElementById('bthVal').textContent=parseFloat(this.value).toFixed(2)"/>
          </div>
          <div class="field">
            <label>PROMPT</label>
            <input type="text" id="promptInput" value="person" placeholder="e.g. person, hard hat"/>
          </div>
          <button class="btn btn-warn btn-full" onclick="pushConfig()">⬆ PUSH CONFIG UPDATE</button>
        </div>
      </div>

      <!-- ROI CONTROLS -->
      <div class="sidebar-section">
        <div class="section-header" onclick="toggleSection(this)">
          <span class="section-title">ROI ZONE EDITOR</span>
          <span>▾</span>
        </div>
        <div class="section-body">
          <p style="font-size:10px;color:var(--muted);margin-bottom:8px">Click on the video feed to place polygon points.</p>
          <div class="roi-info">Points: <span class="roi-pts" id="roiPtsCount">0</span></div>
          <div class="btn-group" style="margin-top:8px">
            <button class="btn btn-success" onclick="pushRoi()">✓ APPLY ROI</button>
            <button class="btn btn-ghost" onclick="clearRoi()">✕ CLEAR</button>
          </div>
          <!-- Evidence thumbnails -->
          <div style="margin-top:12px;font-size:10px;color:var(--muted);letter-spacing:0.08em">EVIDENCE CAPTURES</div>
          <div class="evidence-strip" id="evidenceStrip"></div>
        </div>
      </div>

    </div><!-- /sidebar -->

    <!-- ── CONTENT AREA ─────────────────────────────────────────────── -->
    <div class="content">

      <!-- VIDEO -->
      <div class="video-area" id="videoArea">
        <div class="video-placeholder" id="videoPlaceholder">
          <div class="placeholder-icon">⬡</div>
          <div class="placeholder-text">AWAITING STREAM</div>
        </div>
        <canvas id="videoCanvas" style="display:none"></canvas>
        <canvas id="roiCanvas" style="display:none;position:absolute;top:0;left:0;pointer-events:all" onclick="onVideoClick(event)"></canvas>
        <div class="video-hud" id="videoHud" style="display:none">
          FPS <span id="hudFps2">—</span> &nbsp;|&nbsp; Tracks <span id="hudTracks2">—</span> &nbsp;|&nbsp; Events <span id="hudEvents2">—</span> &nbsp;|&nbsp; Frame <span id="hudFrame2">—</span>
        </div>
      </div>

      <!-- LOG -->
      <div class="log-area">
        <div class="log-header">
          <span class="log-title">SYSTEM LOG</span>
          <button class="btn btn-ghost" style="padding:3px 8px;font-size:10px" onclick="clearLog()">CLEAR</button>
        </div>
        <div class="log-body" id="logBody"></div>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /app -->

<!-- EVIDENCE MODAL -->
<div class="modal-overlay" id="evidenceModal" onclick="closeModal(event)">
  <div class="modal-box">
    <div style="text-align:right"><span class="modal-close" onclick="document.getElementById('evidenceModal').classList.remove('open')">✕</span></div>
    <img class="modal-img" id="modalImg" src="" alt="Evidence"/>
    <div class="modal-meta" id="modalMeta"></div>
  </div>
</div>

<script>
// ── SOCKET.IO ──────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect', () => log('WebSocket connected to web server', 'success'));
socket.on('disconnect', () => log('WebSocket disconnected', 'warn'));
socket.on('log', d => log(d.msg, d.level || 'info'));

socket.on('frame', d => {
  showFrame(d.jpeg, d.meta);
});

socket.on('alert_event', evt => {
  const ts = new Date().toLocaleTimeString();
  log(`──── ALERT ────────────────────────────────────`, 'alert');
  log(`  Rule: ${evt.rule_id}  Track: #${evt.track_id}`, 'alert');
  log(`  Action: ${evt.action}  @ ${parseFloat(evt.timestamp||0).toFixed(2)}s`, 'alert');
  log(`────────────────────────────────────────────────`, 'alert');
});

socket.on('evidence', d => {
  addEvidence(d.jpeg, d.meta);
});

socket.on('status_update', d => {
  updateStatusFromData(d);
});

// ── STATE ──────────────────────────────────────────────────────────────
let sessionRunning = false;
let videoUploaded  = false;
let roiPoints      = [];
let canvasW = 0, canvasH = 0;
let frameW  = 0, frameH  = 0;
let canvasOffX = 0, canvasOffY = 0, canvasScale = 1;

// ── LOG ────────────────────────────────────────────────────────────────
function log(msg, level='info') {
  const body = document.getElementById('logBody');
  const ts = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = `<span class="log-ts">[${ts}]</span><span class="log-msg ${level}">${escHtml(msg)}</span>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  // Keep max 200 lines
  while (body.children.length > 200) body.removeChild(body.firstChild);
}
function clearLog() { document.getElementById('logBody').innerHTML = ''; }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── STATUS ─────────────────────────────────────────────────────────────
function setStatus(label, cls) {
  document.getElementById('statusLabel').textContent = label;
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot ' + (cls||'');
}

function updateStatusFromData(d) {
  const state = d.state || 'idle';
  if (state === 'running') setStatus('LIVE', 'live');
  else if (state === 'loading') setStatus('LOADING', 'loading');
  else if (state === 'error') setStatus('ERROR', 'error');
  else setStatus('IDLE', '');
}

// ── HUD ────────────────────────────────────────────────────────────────
function updateHud(meta) {
  const fps    = meta.fps    !== undefined ? meta.fps    : '—';
  const tracks = meta.tracks !== undefined ? meta.tracks : '—';
  const events = meta.events !== undefined ? meta.events : '—';
  const frame  = meta.frame_id !== undefined ? meta.frame_id : '—';
  ['hudFps','hudFps2'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=fps; });
  ['hudTracks','hudTracks2'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=tracks; });
  ['hudEvents','hudEvents2'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=events; });
  ['hudFrame','hudFrame2'].forEach(id => { const el=document.getElementById(id); if(el) el.textContent=frame; });
}

// ── FRAME RENDERING ────────────────────────────────────────────────────
const videoCanvas = document.getElementById('videoCanvas');
const roiCanvas   = document.getElementById('roiCanvas');
const vCtx = videoCanvas.getContext('2d');
const rCtx = roiCanvas.getContext('2d');
const img  = new Image();

img.onload = function() {
  const area = document.getElementById('videoArea');
  canvasW = area.clientWidth;
  canvasH = area.clientHeight;

  videoCanvas.width  = canvasW;
  videoCanvas.height = canvasH;
  roiCanvas.width    = canvasW;
  roiCanvas.height   = canvasH;

  frameW = img.width;
  frameH = img.height;

  canvasScale = Math.min(canvasW / frameW, canvasH / frameH);
  const nw = frameW * canvasScale;
  const nh = frameH * canvasScale;
  canvasOffX = (canvasW - nw) / 2;
  canvasOffY = (canvasH - nh) / 2;

  vCtx.clearRect(0, 0, canvasW, canvasH);
  vCtx.fillStyle = '#05080d';
  vCtx.fillRect(0, 0, canvasW, canvasH);
  vCtx.drawImage(img, canvasOffX, canvasOffY, nw, nh);

  // Draw track bboxes and ROI from meta stored on img
  if (img._meta) drawAnnotations(img._meta);
  drawRoiOverlay();
};

function showFrame(jpeg, meta) {
  document.getElementById('videoPlaceholder').style.display = 'none';
  videoCanvas.style.display = 'block';
  roiCanvas.style.display   = 'block';
  document.getElementById('videoHud').style.display = 'block';
  updateHud(meta);
  img._meta = meta;
  img.src   = 'data:image/jpeg;base64,' + jpeg;
}

function drawAnnotations(meta) {
  // Already drawn on vCtx; just draw track bboxes over
  const tracks = meta.tracks_data || [];
  tracks.forEach(t => {
    const [x1,y1,x2,y2] = t.bbox;
    const sx1 = canvasOffX + x1 * canvasScale;
    const sy1 = canvasOffY + y1 * canvasScale;
    const sx2 = canvasOffX + x2 * canvasScale;
    const sy2 = canvasOffY + y2 * canvasScale;
    vCtx.strokeStyle = t.is_incident ? '#ff3860' : '#00d4ff';
    vCtx.lineWidth   = t.is_incident ? 3 : 2;
    vCtx.strokeRect(sx1, sy1, sx2-sx1, sy2-sy1);
    vCtx.fillStyle = t.is_incident ? '#ff3860' : '#00d4ff';
    vCtx.font = '10px JetBrains Mono';
    vCtx.fillText(`#${t.track_id} ${t.class_name}`, sx1+2, sy1-4);
  });

  // ROI from meta
  const pts = meta.roi_points || [];
  if (pts.length > 1) {
    vCtx.strokeStyle = 'rgba(0,212,255,0.7)';
    vCtx.lineWidth   = 2;
    vCtx.setLineDash([5,3]);
    vCtx.beginPath();
    pts.forEach(([x,y],i) => {
      const sx = canvasOffX + x * canvasScale;
      const sy = canvasOffY + y * canvasScale;
      i === 0 ? vCtx.moveTo(sx,sy) : vCtx.lineTo(sx,sy);
    });
    vCtx.closePath();
    vCtx.stroke();
    vCtx.setLineDash([]);
  }
}

function drawRoiOverlay() {
  rCtx.clearRect(0, 0, canvasW, canvasH);
  if (roiPoints.length < 2) return;
  rCtx.strokeStyle = '#ffb300';
  rCtx.lineWidth   = 2;
  rCtx.setLineDash([4,3]);
  rCtx.beginPath();
  roiPoints.forEach(([x,y],i) => {
    const sx = canvasOffX + x * canvasScale;
    const sy = canvasOffY + y * canvasScale;
    i === 0 ? rCtx.moveTo(sx,sy) : rCtx.lineTo(sx,sy);
  });
  rCtx.closePath();
  rCtx.stroke();
  rCtx.setLineDash([]);
  roiPoints.forEach(([x,y]) => {
    const sx = canvasOffX + x * canvasScale;
    const sy = canvasOffY + y * canvasScale;
    rCtx.beginPath();
    rCtx.arc(sx, sy, 4, 0, Math.PI*2);
    rCtx.fillStyle = '#ffb300';
    rCtx.fill();
  });
}

// ── VIDEO CLICK (ROI) ──────────────────────────────────────────────────
function onVideoClick(e) {
  if (!frameW) return;
  const rect = roiCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  // Convert canvas px → frame px
  const fx = Math.round((px - canvasOffX) / canvasScale);
  const fy = Math.round((py - canvasOffY) / canvasScale);
  if (fx < 0 || fy < 0 || fx > frameW || fy > frameH) return;
  roiPoints.push([fx, fy]);
  document.getElementById('roiPtsCount').textContent = roiPoints.length;
  log(`ROI point added: [${fx}, ${fy}]`, 'muted');
  drawRoiOverlay();
}

function clearRoi() {
  roiPoints = [];
  document.getElementById('roiPtsCount').textContent = 0;
  rCtx.clearRect(0, 0, canvasW, canvasH);
  log('ROI points cleared', 'warn');
}

// ── SECTION TOGGLE ─────────────────────────────────────────────────────
function toggleSection(header) {
  const body = header.nextElementSibling;
  const arrow = header.querySelector('span:last-child');
  if (!body) return;
  const collapsed = body.classList.toggle('collapsed');
  arrow.textContent = collapsed ? '▸' : '▾';
}

// ── RADIO GROUPS ───────────────────────────────────────────────────────
function onTaskChange(el) {
  document.querySelectorAll('#taskGroup .radio-item').forEach(item => {
    item.classList.toggle('active', item.contains(el));
  });
}

function onSrcChange(el) {
  document.querySelectorAll('#srcGroup .radio-item').forEach(item => {
    item.classList.toggle('active', item.contains(el));
  });
  const val = el.value;
  document.getElementById('srcVideoFile').style.display  = val === 'video_file'    ? '' : 'none';
  document.getElementById('srcJetsonFile').style.display = val === 'jetson_file'   ? '' : 'none';
  document.getElementById('srcCamera').style.display     = val === 'client_camera' ? '' : 'none';
}

// ── TASK YAML TEMPLATES ────────────────────────────────────────────────
const YAML_TEMPLATES = {
  1: `rule_id: rule_roi_entry
description: ROI Zone Entry Detection
detection:
  prompt_positive: person
  box_threshold: 0.25
  text_threshold: 0.25
conditions:
  dwell_seconds: 0
  min_confidence: 0.4
  min_frames: 3
actions:
  cooldown_seconds: 5
roi:
  enabled: true
  type: polygon
  points: []
`,
  2: `rule_id: rule_dwell_time
description: Dwell Time Violation
detection:
  prompt_positive: person
  box_threshold: 0.25
  text_threshold: 0.25
conditions:
  dwell_seconds: 5.0
  min_confidence: 0.4
  min_frames: 3
actions:
  cooldown_seconds: 10
roi:
  enabled: true
  type: polygon
  points: []
`,
  3: `rule_id: rule_helmet_safety
description: Helmet Safety Detection
detection:
  prompt_positive: person without helmet
  box_threshold: 0.25
  text_threshold: 0.25
conditions:
  dwell_seconds: 2.0
  min_confidence: 0.35
  min_frames: 5
  require_helmetless: true
actions:
  cooldown_seconds: 8
roi:
  enabled: false
  type: polygon
  points: []
`,
  4: `rule_id: rule_proximity_danger
description: Proximity Danger Detection
detection:
  prompt_positive: person, forklift
  box_threshold: 0.25
  text_threshold: 0.25
conditions:
  dwell_seconds: 1.0
  min_confidence: 0.4
  min_frames: 3
  near_class: forklift
  near_distance_px: 120
actions:
  cooldown_seconds: 5
roi:
  enabled: false
  type: polygon
  points: []
`,
};

function loadDefaultYaml() {
  const task = document.querySelector('input[name="task"]:checked').value;
  document.getElementById('ruleYaml').value = YAML_TEMPLATES[task] || YAML_TEMPLATES[1];
  log(`Loaded template for task ${task}`, 'info');
}

// ── UPLOAD VIDEO ───────────────────────────────────────────────────────
async function uploadVideo() {
  const fileInput = document.getElementById('videoFile');
  if (!fileInput.files.length) { log('Select a video file first', 'warn'); return; }
  const file = fileInput.files[0];
  const bar  = document.getElementById('uploadProgress');
  const fill = document.getElementById('uploadFill');
  bar.classList.add('active');
  fill.style.width = '0%';
  document.getElementById('btnUpload').disabled = true;
  log(`Uploading ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)...`, 'info');

  const form = new FormData();
  form.append('file', file);
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload_video');
    xhr.upload.onprogress = e => {
      const pct = e.lengthComputable ? Math.round(e.loaded/e.total*100) : 50;
      fill.style.width = pct + '%';
    };
    xhr.onload = () => {
      bar.classList.remove('active');
      document.getElementById('btnUpload').disabled = false;
      if (xhr.status === 200) {
        const r = JSON.parse(xhr.responseText);
        videoUploaded = true;
        log(`Upload complete → ${r.path}`, 'success');
      } else {
        log(`Upload failed: ${xhr.responseText}`, 'error');
      }
    };
    xhr.onerror = () => { log('Upload network error', 'error'); bar.classList.remove('active'); document.getElementById('btnUpload').disabled = false; };
    xhr.send(form);
  } catch(e) { log(`Upload error: ${e}`, 'error'); }
}

// ── START SESSION ──────────────────────────────────────────────────────
async function startSession() {
  const srcVal = document.querySelector('input[name="src"]:checked').value;
  const ruleYaml = document.getElementById('ruleYaml').value.trim();
  if (!ruleYaml) { log('Rule YAML is empty', 'warn'); return; }
  if (srcVal === 'video_file' && !videoUploaded) {
    log('Upload a video file first', 'warn'); return;
  }

  const payload = {
    source_type:   srcVal,
    rule_yaml:     ruleYaml,
    detector:      document.getElementById('detector').value,
    device:        document.getElementById('device').value,
    det_interval:  parseInt(document.getElementById('detInterval').value),
    stream_width:  parseInt(document.getElementById('streamWidth').value),
    jpeg_quality:  parseInt(document.getElementById('jpegQuality').value),
    cam_index:     parseInt(document.getElementById('camIndex').value),
  };
  if (srcVal === 'jetson_file') {
    payload.jetson_file_path = document.getElementById('jetsonFilePath').value.trim();
  }

  log('Starting session...', 'info');
  setStatus('CONNECTING', 'loading');
  document.getElementById('btnStart').disabled = true;

  try {
    const r = await fetch('/api/start', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (r.ok) {
      sessionRunning = true;
      setStatus('LIVE', 'live');
      document.getElementById('btnStop').disabled = false;
      log('Session started ✓', 'success');
      // Sync dwell from rule
      const m = ruleYaml.match(/dwell_seconds:\s*([\d.]+)/);
      if (m) {
        const dw = parseFloat(m[1]);
        document.getElementById('dwellSlider').value = dw;
        document.getElementById('dwellVal').textContent = dw.toFixed(1);
      }
    } else {
      setStatus('ERROR', 'error');
      document.getElementById('btnStart').disabled = false;
      log(`Start failed: ${data.error || JSON.stringify(data)}`, 'error');
    }
  } catch(e) {
    setStatus('ERROR', 'error');
    document.getElementById('btnStart').disabled = false;
    log(`Start error: ${e}`, 'error');
  }
}

// ── STOP SESSION ───────────────────────────────────────────────────────
async function stopSession() {
  log('Stopping session...', 'warn');
  document.getElementById('btnStop').disabled = true;
  try {
    await fetch('/api/stop', { method: 'POST' });
    log('Session stopped', 'warn');
  } catch(e) { log(`Stop error: ${e}`, 'error'); }
  sessionRunning = false;
  setStatus('IDLE', '');
  document.getElementById('btnStart').disabled = false;
  document.getElementById('videoPlaceholder').style.display = '';
  videoCanvas.style.display = 'none';
  roiCanvas.style.display   = 'none';
  document.getElementById('videoHud').style.display = 'none';
}

// ── PUSH CONFIG ────────────────────────────────────────────────────────
async function pushConfig() {
  const payload = {
    dwell_seconds:   parseFloat(document.getElementById('dwellSlider').value),
    min_confidence:  parseFloat(document.getElementById('confSlider').value),
    box_threshold:   parseFloat(document.getElementById('bthSlider').value),
    prompt_positive: document.getElementById('promptInput').value.trim() || undefined,
  };
  log('Pushing config update...', 'info');
  try {
    const r = await fetch('/api/config/update', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    const data = await r.json();
    log(`Config updated: ${JSON.stringify(data.changed || data)}`, 'success');
  } catch(e) { log(`Config error: ${e}`, 'error'); }
}

// ── PUSH ROI ───────────────────────────────────────────────────────────
async function pushRoi() {
  if (!roiPoints.length) { log('No ROI points defined. Click on video to add.', 'warn'); return; }
  const payload = { roi_points: roiPoints };
  log(`Applying ROI with ${roiPoints.length} points...`, 'info');
  try {
    const r = await fetch('/api/config/update', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
    });
    const data = await r.json();
    log(`ROI applied: ${JSON.stringify(data.changed || data)}`, 'success');
  } catch(e) { log(`ROI error: ${e}`, 'error'); }
}

// ── CHECK STATUS ───────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    log(JSON.stringify(data, null, 2), 'muted');
    updateStatusFromData(data);
  } catch(e) { log(`Status error: ${e}`, 'error'); }
}

// ── EVIDENCE ───────────────────────────────────────────────────────────
function addEvidence(jpeg, meta) {
  const strip = document.getElementById('evidenceStrip');
  const thumb = document.createElement('div');
  thumb.className = 'evidence-thumb';
  const src = 'data:image/jpeg;base64,' + jpeg;
  thumb.innerHTML = `<img src="${src}" alt="evidence"/><div class="ev-badge">#${meta.track_id||'?'}</div>`;
  thumb.onclick = () => openModal(src, meta);
  strip.insertBefore(thumb, strip.firstChild);
  // Max 20 thumbs
  while (strip.children.length > 20) strip.removeChild(strip.lastChild);
}

function openModal(src, meta) {
  document.getElementById('modalImg').src = src;
  document.getElementById('modalMeta').textContent = JSON.stringify(meta, null, 2);
  document.getElementById('evidenceModal').classList.add('open');
}

function closeModal(e) {
  if (e.target === document.getElementById('evidenceModal')) {
    document.getElementById('evidenceModal').classList.remove('open');
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global JETSON_BASE_URL, _relay_stop

    parser = argparse.ArgumentParser(description="OVD Watchdog — Flask Web App")
    parser.add_argument("--host",   default="0.0.0.0", help="Bind host")
    parser.add_argument("--port",   type=int, default=5000, help="Web server port")
    parser.add_argument("--jetson", default="http://localhost:8765", help="Jetson GPU server URL")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    JETSON_BASE_URL = args.jetson
    _relay_stop = threading.Event()

    print("=" * 62)
    print("  OVD WATCHDOG — WEB APPLICATION")
    print("=" * 62)
    print(f"  Web UI   : http://{args.host}:{args.port}")
    print(f"  Jetson   : {JETSON_BASE_URL}")
    print()
    print("  Open in browser: http://localhost:5000")
    print("=" * 62)

    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
