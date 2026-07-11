# 🪸 Web Detectie — Coral AI Mail Carrier Detection System

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-orange?logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/petervandersanden)

RTSP camera video detection system based on a **Coral AI Edge TPU**, which detects people (e.g. the mail carrier) at the mailbox using a state machine, and publishes status to **MQTT** (Home Assistant), sends via **email**, and displays via a built-in **MJPEG web stream**.

> ☕ If this project is useful to you, consider [buying me a coffee](https://buymeacoffee.com/petervandersanden).

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Mail carrier state machine](#mail-carrier-state-machine)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration (config.ini)](#configuration-configini)
- [Usage](#usage)
- [MQTT payload](#mqtt-payload)
- [Web interface](#web-interface)
- [Debug overlay](#debug-overlay)
- [Key implementation details](#key-implementation-details)

## Features

- Reads an RTSP camera stream (e.g. Hikvision) in a separate thread with a **watchdog** (automatic reconnect on stall).
- Performs object detection with a **Coral Edge TPU** model (via `pycoral`), filtered on `target_labels` from the config (e.g. `person`).
- Tracks objects across frames with a lightweight **centroid tracker** (own ID per object, no heavy re-ID models).
- Recognizes the **mail carrier visit pattern** (arrival → mailbox → departure) via a separate state machine per person, with false-positive filters (exclude zone, direction check, dwell time).
- Distinguishes **day/night** based on average image brightness and uses a lower confidence threshold and detection interval at night.
- Publishes detections + snapshot to **MQTT** (retained), with throttling via `mqtt.interval`.
- Optionally sends an **email with photo** as soon as a delivery is confirmed.
- Shows a live **MJPEG video feed with debug overlay** via a built-in Flask/Waitress web server (`/video_feed`, `/`).
- All MQTT and mail actions run through a shared `ThreadPoolExecutor` so the detection loop never blocks.

## Architecture

```
RTSPReader (thread)          detect_objects() (thread)         Flask/Waitress (main thread)
  └─ reads camera frames  →    ├─ Coral TPU inference               └─ /video_feed (MJPEG stream)
     with watchdog restart      ├─ CentroidTracker (ID assignment)   └─ / (HTML preview page)
                                ├─ PostTracker (state machine)
                                ├─ MQTT publish (async, executor)
                                └─ Email on confirmation (async, executor)
```

- **`RTSPReader`** — dedicated thread that continuously reads frames (throttled to `reader_fps`), with a **watchdog thread** that restarts the stream if no new frame arrives for more than `WATCHDOG_SECONDS` (8s). Prevents duplicate RTSP sessions by cleanly waiting for the old thread (`join()`) before starting a new one.
- **`CentroidTracker`** — matches detections across frames based on label + distance between centroids, assigns/removes IDs after `max_disappeared` missed frames, and fully resets itself if nothing has been seen for more than `reset_seconds`.
- **`PostTracker`** — standalone state machine per object ID that recognizes the delivery pattern (see below).
- **Detection loop** (`detect_objects`) — the main thread that fetches frames, runs inference, updates tracking/state, draws the debug overlay, and triggers MQTT/mail. Throttled to `detect_fps`.
- **Web server** — Flask app served via **Waitress** (production WSGI server), shows the latest frame with overlay as an MJPEG stream.

## Mail carrier state machine

All zone coordinates are in **AI coordinates (0–300)**, the resolution of the detection model.

| State | Meaning |
|---|---|
| `IDLE` | Rest state, waiting for arrival from the right |
| `APPROACHING` | Person moving from the right toward the mailbox |
| `AT_MAILBOX` | Person is located in the mailbox zone |
| `DEPARTING` | Person is leaving the mailbox, moving right |
| `POST_CONFIRMED` | Full pattern confirmed → mail delivered |

**Three hard requirements for `POST_CONFIRMED`:**

1. **Arrival** — track starts top-right (`cx > arrive_min_cx`, `cy < arrive_max_cy`), not in the exclude zone, and then moves toward the mailbox (for `approach_min_seconds`).
2. **Delivery** — person is in the `mailbox_zone` for between `mailbox_dwell_min` and `mailbox_dwell_max` seconds.
3. **Departure** — person leaves the mailbox and moves back to the right (`cx > depart_min_cx`).

**False-positive filters:**
- **Exclude zone** top-right ignores passersby not walking toward the mailbox.
- **Direction requirement** on arrival and departure (cx must decrease/increase), based on the last 3 positions in the movement history.
- **Max dwell** — standing at the mailbox too long is not treated as the mail carrier (reset).
- If a person disappears from view during `DEPARTING` (e.g. outside camera range) at the correct position, the delivery is still confirmed (`pending_post_ids`) once the tracker cleans up the object.
- After confirmation, a **cooldown** (`post_cooldown`) applies before the same ID can trigger again.

## Requirements

- Python 3 with:
  - `opencv-python` (`cv2`)
  - `numpy`
  - `paho-mqtt` (v2 callback API)
  - `flask`
  - `waitress`
  - `pycoral` (Coral Edge TPU runtime + libraries)
- **Coral USB/PCIe Edge TPU** with a compiled `.tflite` model + label file.
- RTSP camera (e.g. Hikvision) reachable on the network.
- MQTT broker (e.g. Mosquitto, or the Home Assistant broker).
- (Optional) SMTP account for email notifications.

## Installation

```bash
pip install opencv-python numpy paho-mqtt flask waitress
# Install Coral/pycoral according to the official Coral documentation
# (Edge TPU runtime + pycoral libraries, depending on your platform)
```

Place `config.ini` in the same folder as `mailbox_detection.py` (see below).

> **Note:** the `config.ini` section/key names (`[detectie]`, `[post_detectie]`, etc.) and the internal label check `'persoon'` were intentionally left as-is, since they must keep matching your existing `config.ini` and Edge TPU label file. Let me know if you'd like those renamed to English too — that would also require updating `config.ini` and the label file to match.

## Configuration (config.ini)

The script reads all settings from a `config.ini` next to the script file. Example structure:

```ini
[camera]
model_path = /path/to/model.tflite
label_path = /path/to/labels.txt
rtsp_url = rtsp://user:pass@camera-ip:554/stream
reader_fps = 15
detect_fps = 8

[mqtt]
broker = 192.168.1.10
port = 1883
user = mqtt_user
pw = mqtt_password
topic = detectie/brievenbus
interval = 2.0

[mail]
enabled = true
smtp_server = smtp.example.com
smtp_port = 587
from_addr = maildetection@example.com
from_name = Mail Detection
to_addr = you@example.com

[detectie]
min_confidence_day = 0.55
min_confidence_night = 0.45
night_interval = 5.0
target_labels = person

[post_detectie]
mailbox_zone = 78,130,102,188
exclude_zone = 200,0,300,70
arrive_min_cx = 220
arrive_max_cy = 60
depart_min_cx = 220
depart_max_cy = 60
mailbox_dwell_min = 1.5
mailbox_dwell_max = 15
approach_min_seconds = 1.0
post_cooldown = 300

[tracker]
max_disappeared = 15
reset_seconds = 60
max_trace_points = 30

[debug]
draw_debug = true
draw_traces_debug = true
```

> ⚠️ The values above are illustrative — adjust the zone coordinates based on your own camera view (AI coordinates run from 0–300, regardless of the actual camera resolution).

**Zone format:** `x1,y1,x2,y2` in AI coordinates (0–300), where (0,0) is top-left.

## Usage

```bash
python3 mailbox_detection.py
```

- The detection loop starts automatically in a background thread.
- The web server runs on port **5000** (`0.0.0.0:5000`), via Waitress with 6 threads.
- Live preview: `http://<host>:5000/`
- Raw MJPEG stream: `http://<host>:5000/video_feed`

## MQTT payload

On every detection (max. once per `mqtt.interval` seconds), the following is published to the configured topic:

```json
{
  "timestamp": "14:32:10",
  "label": "person",
  "id": 3,
  "score": 87,
  "is_post": false,
  "post_state": "APPROACHING",
  "is_new_detection": true,
  "is_night": false,
  "detections": [ /* array of all detected objects in this frame */ ]
}
```

In addition, a JPEG snapshot (retained) is published to `<topic>/snapshot` on every MQTT update.

Once `is_post: true` and `post_state: "POST_CONFIRMED"`, the delivery is confirmed — this is also the moment (if enabled) that the email with snapshot is sent.

## Web interface

- **`/`** — simple HTML page with the live stream and a color legend:
  - Gray = `IDLE`
  - Yellow = `APPROACHING`
  - Orange = `AT_MAILBOX`
  - Red = `POST_CONFIRMED`
- **`/video_feed`** — multipart MJPEG stream (usable as a camera source in Home Assistant or an `<img>` tag).

## Debug overlay

If `debug.draw_debug = true`:
- The mailbox zone and exclude zone are drawn as colored areas.
- Arrival/departure lines (`arrive_min_cx` / `arrive_max_cy`) as reference lines.
- Bounding box + ID + label + state per detected object, colored by current state.
- "POST DELIVERED!" text upon confirmation.

If `debug.draw_traces_debug = true`:
- Movement traces (polylines) per object ID, colored by state.

At night, a "NIGHT" indicator appears in the top-right of the frame.

## Key implementation details

- **Threading architecture**: camera reading, inference/tracking, and the web server run in separate threads; MQTT publishing and email sending run through a shared `ThreadPoolExecutor(max_workers=2)` so the detection loop never blocks on network I/O.
- **Paho MQTT v2**: client is created with `CallbackAPIVersion.VERSION2`.
- **Throttling**: separate FPS limits for reading the camera (`reader_fps`, default 15) and the detection/inference loop (`detect_fps`, default 8), configurable via `config.ini`.
- **RTSP stability**: the watchdog detects a stalled stream (no new frame for > 8s) and restarts the reader thread only after confirmed shutdown of the old thread, to prevent duplicate connections to the camera (important given Hikvision's connection limit).
- **AI vs. pixel coordinates**: all zone logic works in AI coordinates (0–300, the model input format); `ai_to_px()` / `ai_zone_to_px()` convert this to pixels for the overlay on the full camera image.
- **Exception isolation**: the main loop catches and logs errors per iteration, so an occasional error (e.g. a camera hiccup) doesn't crash the whole service.

---

☕ **Enjoying this project?** Support further development via [Buy Me a Coffee](https://buymeacoffee.com/petervandersanden).

*This README.md was automatically generated based on analysis of `mailbox_detection.py`. Fill in/adjust the config.ini example values with your own camera, MQTT, and mail settings.*
