# OVD Watchdog System

## Detection + Tracking Only Verification

This deliverable provides a minimal pipeline integrating **GroundingDINO** text-prompt detection and **ByteTrack** multi-object tracking. It visualizes bounding boxes and track IDs on MP4 or camera inputs and can optionally export video/JSON outputs. It allows validation of detection and tracking behavior only, without using rule engines or notification modules.

### Purpose

The verification tool allows you to:
- Test GroundingDINO detection accuracy with custom text prompts
- Verify ByteTrack tracking stability and ID consistency across frames
- Quick validation on new videos or camera feeds
- Export results for analysis (video + JSON)

### Quick Start

#### 1. Setup
```bash
# Install dependencies (install cuda fit with your env by yourself its not include in requirements.txt)
pip install -r requirements.txt
```

#### 2. Run Verification

**MP4 File (CPU mode)**
```bash
python tools/run_det_track.py --input data/test_videos/sample.mp4 --prompt "person" --device cpu
```

**USB Camera (GPU mode)**
```bash
python tools/run_det_track.py --input 0 --prompt "person" --device cuda
```

**With Video Export**
```bash
python tools/run_det_track.py --input data/test_videos/sample.mp4 --prompt "person" --output outputs/tracked_video.mp4 --json --device cuda
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Video file path or camera index (0, 1, ...) | Required |
| `--prompt` | Detection prompts (space-separated) | `"person"` |
| `--device` | Device to run on (`cuda` or `cpu`) | `cuda` |
| `--output` | Output video path | None |
| `--json` | Export tracking data to JSON | False |
| `--box-threshold` | Detection confidence threshold | 0.35 |
| `--detection-interval` | Detect every N frames | 5 |

### Expected Output

#### Console Output
```
======================================================================
🔍 DETECTION + TRACKING VERIFICATION
======================================================================
Input: data/test_videos/sample.mp4
Prompt: person
Device: cuda
Detection interval: 5 frames
======================================================================

[1/3] Initializing video source...
✓ Video file opened: data/test_videos/sample.mp4

[2/3] Loading detector...
✓ GroundingDINO loaded on cuda

[3/3] Initializing tracker...
✓ ByteTrack initialized

======================================================================
▶️  PROCESSING STARTED (Press 'q' to quit, 'p' to pause)
======================================================================

Frame 5: Detected 3 objects
Frame 100: Detected 2 objects
Processed 100 frames | FPS: 15.3 | Tracks: 3
Processed 200 frames | FPS: 16.8 | Tracks: 2

======================================================================
✓ PROCESSING COMPLETE
======================================================================

Summary:
  Frames processed: 300
  Total tracks created: 5
  Average FPS: 16.2
  ✓ JSON exported: outputs/tracked_video.json
  ✓ Video saved: outputs/tracked_video.mp4
```

#### Visualization Window

The window shows:
- **Bounding boxes** with consistent colors per track ID
- **Track ID** and **state** (tentative/confirmed)
- **Confidence score** for each detection
- **Frame info**: Total tracks, detections, FPS

**Example:**
```
┌─────────────────────────────────────┐
│ Tracks: 3                           │
│ Detections: 3                       │
│ Frame: 150                  FPS: 16.2│
│                                     │
│  ┌──────────────┐                  │
│  │ ID:1 [confirmed] 0.85           │
│  │              │                  │
│  └──────────────┘                  │
│         ┌──────────┐               │
│         │ ID:2 [confirmed] 0.92    │
│         │          │               │
│         └──────────┘               │
└─────────────────────────────────────┘
```

#### JSON Output Format
```json
[
  {
    "frame_id": 1,
    "timestamp": 0.033,
    "tracks": [
      {
        "track_id": 1,
        "bbox": [120.5, 200.3, 350.2, 500.8],
        "confidence": 0.85,
        "class": "person",
        "state": "confirmed"
      }
    ]
  },
  {
    "frame_id": 2,
    "timestamp": 0.066,
    "tracks": [
      {
        "track_id": 1,
        "bbox": [125.1, 205.6, 355.3, 505.2],
        "confidence": 0.87,
        "class": "person",
        "state": "confirmed"
      }
    ]
  }
]
```